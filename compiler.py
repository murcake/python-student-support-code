from __future__ import annotations

import ast as py
import sys
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from itertools import chain
from types import MappingProxyType
from typing import Generic, TypeVar

import x86_ast as x86
from priority_queue import PriorityQueue
from utils import Begin as PyBegin
from utils import CProgram
from utils import Goto as CGoto
from utils import label_name
from x86_ast import X86Program, location


class CompilerError(Exception):
    pass


class UnsupportedNode(CompilerError):
    def __init__(self, node: py.AST):
        msg = f"The following node {(type(node))} is not supported by the language: {repr(node)}"
        super().__init__(msg)


class NonAtomicExpr(CompilerError):
    def __init__(self, e: py.expr):
        msg = f"The following expression is not atomic: {e}"
        super().__init__(msg)


class TooManyArgsForInstr(CompilerError):
    def __init__(self, i: x86.Instr):
        msg = f"The following instruction has too many (> 2) arguments: {i}"
        super().__init__(msg)


class MissingPass(CompilerError):
    def __init__(self, pass_name: str):
        msg = f"Complete the following pass first: f{pass_name}"
        super().__init__(msg)


class CallqArityTooBig(CompilerError):
    def __init__(self, func: str, arity: int):
        threshold = len(arg_passing_regs)
        super().__init__(
            f"Arity of a function call (to '{func}') is too big ({arity} > {threshold})"
        )


class UnsupportedInstr(CompilerError):
    def __init__(self, i: x86.instr):
        msg = f"The following instruction is not supported: {i}"
        super().__init__(msg)


class PartialEvaluator:
    def __init__(self):
        self.__constant_vars: dict[py.Name, py.Constant] = {}

    def run(self, p: py.Module) -> py.Module:
        return self._visit_module(p)

    def _visit_module(self, p: py.Module) -> py.Module:
        body = []
        for stmt_ in p.body:
            if stmt := self._visit_stmt(stmt_):
                body.append(stmt)
        return py.Module(body)

    def _visit_stmt(self, stmt: py.stmt) -> None | py.stmt:
        match stmt:
            case py.Assign([py.Name() as lhs], value_):
                value = self._visit_expr(value_)
                if isinstance(value, py.Constant):
                    self.__constant_vars[lhs] = value
                return py.Assign([lhs], value)
            case py.Expr(py.Call(py.Name("print"), args_)):
                args = [self._visit_expr(x) for x in args_]
                return py.Expr(py.Call(py.Name("print"), args))
            case py.Expr(expr_):
                expr = self._visit_expr(expr_)
                if isinstance(expr, py.Constant):
                    return None
                return py.Expr(expr)
            case _:
                raise UnsupportedNode(stmt)

    def _visit_expr(self, expr: py.expr) -> py.expr:
        match expr:
            case py.Constant() | py.Call(py.Name("input_int")):
                return expr
            case py.Name():
                return self.__constant_vars.get(expr, expr)
            case py.UnaryOp(py.USub(), arg):
                return self._visit_usub(self._visit_expr(arg))
            case py.BinOp(arg1_, py.Sub(), arg2_):
                arg1 = self._visit_expr(arg1_)
                arg2 = self._visit_expr(py.UnaryOp(py.USub(), arg2_))
                return self._visit_add(arg1, arg2)
            case py.BinOp(arg1_, py.Add(), arg2_):
                return self._visit_add(self._visit_expr(arg1_), self._visit_expr(arg2_))
            case _:
                raise UnsupportedNode(expr)

    def _visit_usub(self, expr: py.expr) -> py.expr:
        match expr:
            case py.Constant(v):
                return py.Constant(-v)
            case py.UnaryOp(py.USub(), x):
                return x
            case py.BinOp(x, py.Add(), y):
                return py.BinOp(
                    py.UnaryOp(py.USub(), x), py.Add(), py.UnaryOp(py.USub(), y)
                )
            case _:
                return py.UnaryOp(py.USub(), expr)

    def _visit_add(self, x: py.expr, y: py.expr) -> py.expr:
        match (x, y):
            case (a, py.Constant(0)):
                return a
            case (py.Constant(a), py.Constant(c)):
                return py.Constant(a + c)
            case (py.Constant() as a, c):
                return self._visit_add(c, a)  # safe, a is side-effect free
            case (py.BinOp(a, py.Add(), py.Constant(b)), py.Constant(c)):
                return self._visit_add(a, py.Constant(b + c))
            case (a, py.BinOp(c, py.Add(), py.Constant() as d)):
                return py.BinOp(py.BinOp(a, py.Add(), c), py.Add(), d)
            case (a, c):
                return py.BinOp(a, py.Add(), c)


class PartialEvaluatorIf(PartialEvaluator):
    def _visit_stmt(self, stmt: py.stmt) -> py.stmt | None:
        match stmt:
            case py.If():
                return stmt
            case _:
                return super()._visit_stmt(stmt)

    def _visit_expr(self, expr: py.expr) -> py.expr:
        match expr:
            case py.IfExp() | py.BoolOp() | py.Compare():
                return expr
            case _:
                return super()._visit_expr(expr)


class RemoveComplexOperands:
    def __init__(self):
        self.tmp_cnt: int = 0
        self.tmp_bindings: list[tuple[py.Name, py.expr]] = []

    def run(self, p: py.Module) -> py.Module:
        return self._visit_module(p)

    def _visit_module(self, p: py.Module) -> py.Module:
        return py.Module(list(self._visit_stmts(p.body)))

    def _visit_stmts(self, stmts: Iterable[py.stmt]) -> Iterable[py.stmt]:
        for stmt in stmts:
            yield from self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: py.stmt) -> Iterable[py.stmt]:
        with self._new_scope():
            main = self._visit_stmt_meat(stmt)
            yield from (py.Assign([n], v) for (n, v) in self.tmp_bindings)
            yield main

    def _visit_stmt_meat(self, stmt: py.stmt) -> py.stmt:
        match stmt:
            case py.Assign([lhs], value_):
                return py.Assign([lhs], self._visit_expr(value_, mk_atomic=False))
            case py.Expr(py.Call(py.Name("print"), args_)):
                args = [self._visit_expr(arg, mk_atomic=True) for arg in args_]
                return py.Expr(py.Call(py.Name("print"), args))
            case py.Expr(expr):
                return py.Expr(self._visit_expr(expr, mk_atomic=False))
            case _:
                raise UnsupportedNode(stmt)

    def _visit_expr(self, expr: py.expr, mk_atomic: bool) -> py.expr:
        main = self._visit_expr_meat(expr, mk_atomic)
        if mk_atomic and not isinstance(expr, (py.Name, py.Constant)):
            main = self._bind_to_tmp(main)
        return main

    def _visit_expr_meat(self, expr: py.expr, _: bool) -> py.expr:
        match expr:
            case py.Name() | py.Constant():
                return expr
            case py.Call(py.Name("input_int")):
                return expr
            case py.UnaryOp(py.USub(), arg):
                return py.UnaryOp(py.USub(), self._visit_expr(arg, mk_atomic=True))
            case py.BinOp(arg1_, (py.Add() | py.Sub()) as op, arg2_):
                arg1 = self._visit_expr(arg1_, mk_atomic=True)
                arg2 = self._visit_expr(arg2_, mk_atomic=True)
                return py.BinOp(arg1, op, arg2)
            case _:
                raise UnsupportedNode(expr)

    def _bind_to_tmp(self, value: py.expr) -> py.Name:
        name = py.Name(f"$tmp_{self.tmp_cnt}")
        self.tmp_cnt += 1
        self.tmp_bindings.append((name, value))
        return name

    def _get_tmp_assignments(self) -> Iterable[py.Assign]:
        return (py.Assign([n], v) for (n, v) in self.tmp_bindings)

    @contextmanager
    def _new_scope(self):
        old_tmp_bindings = self.tmp_bindings
        self.tmp_bindings = []
        try:
            yield None
        finally:
            self.tmp_bindings = old_tmp_bindings


pyTrue = py.Constant(value=True)
pyFalse = py.Constant(value=False)


class RemoveComplexOperandsIf(RemoveComplexOperands):
    def _visit_stmt_meat(self, stmt: py.stmt) -> py.stmt:
        match stmt:
            case py.If(test_, body_, orelse_):
                test = self._visit_expr(test_, mk_atomic=False)
                body = list(self._visit_stmts(body_))
                orelse = list(self._visit_stmts(orelse_))
                return py.If(test, body, orelse)
            case _:
                return super()._visit_stmt_meat(stmt)

    def _visit_expr_meat(self, expr: py.expr, mk_atomic: bool) -> py.expr:
        match expr:
            case py.IfExp(test_, body_, orelse_):
                test = self._visit_expr(test_, mk_atomic=False)
                with self._new_scope():
                    body_expr = self._visit_expr(body_, mk_atomic=mk_atomic)
                    assigns = self._get_tmp_assignments()
                    body = PyBegin(list(assigns), body_expr)
                with self._new_scope():
                    orelse_expr = self._visit_expr(orelse_, mk_atomic=mk_atomic)
                    assigns = self._get_tmp_assignments()
                    orelse = PyBegin(list(assigns), orelse_expr)
                return py.IfExp(test, body, orelse)
            case py.UnaryOp(py.Not(), arg):
                return py.UnaryOp(py.Not(), self._visit_expr(arg, mk_atomic=True))
            case py.BoolOp(py.And(), [a, b]):
                as_if = py.IfExp(test=a, body=b, orelse=pyFalse)
                return self._visit_expr_meat(as_if, mk_atomic=mk_atomic)
            case py.BoolOp(py.Or(), [a, b]):
                as_if = py.IfExp(test=a, body=pyTrue, orelse=b)
                return self._visit_expr_meat(as_if, mk_atomic=mk_atomic)
            case py.Compare(a_, [op], [b_]):
                a = self._visit_expr(a_, mk_atomic=True)
                b = self._visit_expr(b_, mk_atomic=True)
                return py.Compare(a, [op], [b])
            case _:
                return super()._visit_expr_meat(expr, mk_atomic)


class Block:
    name: str
    stmts: list[py.stmt]

    def __init__(self, name: str, stmts: list[py.stmt] | None = None):
        self.name = name
        self.stmts = stmts or []

    def _as_jump(self) -> list[py.stmt]:
        match self.stmts:
            case [CGoto()]:
                return self.stmts
        return [CGoto(self.name)]

    def link_to(self, b: Block) -> None:
        if self.stmts and isinstance(self.stmts[-1], CGoto):
            return
        self.stmts.extend(b._as_jump())


class ExplicateControl:
    def __init__(self):
        self.current_block: Block = Block(start_label)
        self.blocks: dict[str, list[py.stmt]] = {start_label: self.current_block.stmts}
        self.block_cnt: int = 0

    def run(self, p: py.Module) -> CProgram:
        self._visit_module(p)
        self.current_block.stmts.append(py.Return(py.Constant(value=0)))
        return CProgram(self.blocks)

    def _visit_module(self, p: py.Module):
        self._visit_stmts(p.body)

    def _visit_stmts(self, stmts: Iterable[py.stmt]):
        for stmt in stmts:
            self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: py.stmt) -> None:
        match stmt:
            case py.Assign([py.Name() as lhs], value_):
                self._visit_assign(lhs, value_)
            case py.If(test, body_, orelse_):
                self._visit_if(test, body_, orelse_)
            case _:
                self._emit(stmt)

    def _visit_assign(self, lhs: py.Name, value: py.expr) -> None:
        match value:
            case py.IfExp(test, PyBegin(body_, body_r), PyBegin(orelse_, orelse_r)):
                # breakpoint()
                body_assign = py.Assign([lhs], body_r)
                orelse_assign = py.Assign([lhs], orelse_r)
                self._visit_if(test, body_ + [body_assign], orelse_ + [orelse_assign])
            case _:
                self._emit(py.Assign([lhs], value))

    def _visit_if(self, test: py.expr, body_: list[py.stmt], orelse_: list[py.stmt]):
        original_block = self.current_block
        body_block = self._mk_block()
        orelse_block = self._mk_block()
        next_block = self._mk_block()

        self._continue_with(body_block)
        self._visit_stmts(body_)
        self.current_block.link_to(next_block)

        self._continue_with(orelse_block)
        self._visit_stmts(orelse_)
        self.current_block.link_to(next_block)

        self._continue_with(original_block)
        self._visit_if_blocks(test, body_block, orelse_block)
        self._continue_with(next_block)

    def _visit_if_blocks(self, test, body: Block, orelse: Block):
        match test:
            case py.Constant(value=bool(v)):
                self.current_block.link_to(body if v else orelse)
            case py.Name() as n:
                self._visit_if_blocks(py.Compare(n, [py.Eq()], [pyTrue]), body, orelse)
            case py.Compare():
                self._emit(py.If(test, body._as_jump(), orelse._as_jump()))
            case py.UnaryOp(py.Not(), not_test):
                self._visit_if_blocks(not_test, orelse, body)
            case py.IfExp(i_test, i_body, i_orelse):
                body_link = body._as_jump()
                orelse_link = orelse._as_jump()
                lifted_body: list[py.stmt] = [py.If(i_body, body_link, orelse_link)]
                lifted_orelse: list[py.stmt] = [py.If(i_orelse, body_link, orelse_link)]
                self._visit_if(i_test, lifted_body, lifted_orelse)
            case PyBegin(stmts, inner_test):
                self._visit_stmts(stmts)
                self._visit_if_blocks(inner_test, body, orelse)
            case _:
                raise UnsupportedNode(test)

    def _emit(self, stmt: py.stmt):
        self.current_block.stmts.append(stmt)

    def _continue_with(self, block: Block):
        self.current_block = block

    def _mk_block(self) -> Block:
        name = self._mk_fresh_block_name()
        block = Block(name)
        self.blocks[name] = block.stmts
        return block

    def _mk_fresh_block_name(self) -> str:
        self.block_cnt += 1
        return label_name(f"block_{self.block_cnt}")


class X86Builder:
    def __init__(self):
        self.__current_block: list[x86.instr] | None = None
        self.__x86_body: dict[str, list[x86.instr]] = {}

    def _emit(self, i: x86.instr):
        assert self.__current_block is not None, "called outside of '_new_block'"
        self.__current_block.append(i)

    def _build(self) -> X86Program:
        return X86Program(self.__x86_body)

    @contextmanager
    def _new_block(self, name: str):
        old_current_block = self.__current_block
        self.__current_block = []
        dict_set_fresh(self.__x86_body, name, self.__current_block)
        try:
            yield None
        finally:
            self.__current_block = old_current_block


class SelectInstructions(X86Builder):
    def run(self, p: py.Module) -> X86Program:
        self._visit_module(p)
        return self._build()

    def _visit_module(self, p: py.Module):
        for stmt in p.body:
            self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: py.stmt):
        match stmt:
            case py.Assign([py.Name(lhs)], value):
                self._visit_assign(x86.Variable(lhs), value)
            case py.Expr(py.Call(py.Name("print"), [arg_])):
                arg = take_atomic_arg(arg_)
                self._emit(movq(arg, rdi))
                self._emit(callq_print_int)
            case py.Expr(e):
                self._visit_for_side_effects(e)
            case _:
                raise UnsupportedNode(stmt)

    def _visit_assign(self, destination: x86.arg, value: py.expr):
        if (source := try_take_atomic_arg(value)) is not None:
            self._emit(movq(source, destination))
            return
        match value:
            case py.Call(py.Name("input_int")):
                self._emit(callq_read_int)
                self._emit(movq(rax, destination))
            case py.UnaryOp(py.USub(), arg_):
                arg = take_atomic_arg(arg_)
                if arg != destination:
                    self._emit(movq(arg, destination))
                self._emit(negq(destination))
            case py.BinOp(arg1_, op, arg2_):
                arg1 = take_atomic_arg(arg1_)
                arg2 = take_atomic_arg(arg2_)
                self._visit_operator(op, arg1, arg2, destination)
            case _:
                raise UnsupportedNode(value)

    def _visit_for_side_effects(self, expr: py.expr):
        match expr:
            case py.Call(py.Name("input_int")):
                self._emit(callq_read_int)

    def _visit_operator(
        self, op: py.operator, arg1: x86.arg, arg2: x86.arg, dst: x86.arg
    ):
        match op:
            case py.Sub():
                if arg1 != dst:
                    self._emit(movq(arg1, dst))
                self._emit(subq(arg2, dst))
            case py.Add():
                if arg2 == dst:
                    arg1, arg2 = arg2, arg1  # commutativity
                if arg1 != dst:
                    self._emit(movq(arg1, dst))
                self._emit(addq(arg2, dst))
            case _:
                raise UnsupportedNode(op)


class SelectInstructionsIf(SelectInstructions):
    def __init__(self):
        super().__init__()
        self.side_effect_only: bool = False

    def run(self, cprogram: CProgram) -> X86Program:
        self._visit_cprogram(cprogram)
        return super()._build()

    def _visit_cprogram(self, cprogram: CProgram):
        for (label, stmts) in cprogram.body.items():
            with self._new_block(label):
                for stmt in stmts:
                    self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: py.stmt):
        match stmt:
            case py.If(py.Compare(a_, [op], [b_]), [CGoto(body)], [(CGoto(orelse))]):
                a = take_atomic_arg(a_)
                b = take_atomic_arg(b_)
                self._emit(cmpq(lhs=a, rhs=b))
                self._emit(x86.JumpIf(get_compare_kind(op), body))
                self._emit(x86.Jump(orelse))
            case py.Return(py.expr() as v_):
                v = take_atomic_arg(v_)
                self._emit(movq(v, rax))
                self._emit(x86.Jump(conclusion_label))
                with self._new_block(conclusion_label):
                    pass  # just creating the block
            case CGoto(label):
                self._emit(x86.Jump(label))
            case _:
                super()._visit_stmt(stmt)

    def _visit_assign(self, destination: x86.arg, value: py.expr):
        match value:
            case py.UnaryOp(py.Not(), arg_):
                arg = take_atomic_arg(arg_)
                if arg != destination:
                    self._emit(movq(arg, destination))
                self._emit(xorq(imm1, destination))
            case py.Compare(a_, [op], [b_]):
                a = take_atomic_arg(a_)
                b = take_atomic_arg(b_)
                self._emit(cmpq(lhs=a, rhs=b))
                self._emit(x86_set(op))
                self._emit(movzbq_al(destination))
            case _:
                return super()._visit_assign(destination, value)

    def _visit_for_side_effects(self, expr: py.expr):
        with self._side_effect_only():
            match expr:
                case py.IfExp(_, body, orelse):
                    self._visit_for_side_effects(body)
                    self._visit_for_side_effects(orelse)
                case PyBegin(stmts, e):
                    for stmt in stmts:
                        self._visit_stmt(stmt)
                    self._visit_for_side_effects(e)
                case _:
                    super()._visit_for_side_effects(expr)

    @contextmanager
    def _side_effect_only(self):
        backup = self.side_effect_only
        self.side_effect_only = True
        try:
            yield None
        finally:
            self.side_effect_only = backup

    def _emit(self, i: x86.instr):
        if self.side_effect_only and not isinstance(i, x86.Callq):
            return None
        super()._emit(i)


def try_take_atomic_arg(e: py.expr) -> None | x86.arg:
    match e:
        case py.Name(n):
            return x86.Variable(n)
        case py.Constant(v):
            return x86.Immediate(int(v))
        case _:
            return None


def take_atomic_arg(e: py.expr) -> x86.arg:
    if res := try_take_atomic_arg(e):
        return res
    else:
        raise NonAtomicExpr(e)


def get_compare_kind(op: py.cmpop) -> str:
    if isinstance(op, py.Eq):
        return "e"
    elif isinstance(op, py.NotEq):
        return "ne"
    elif isinstance(op, py.Gt):
        return "g"
    elif isinstance(op, py.GtE):
        return "ge"
    elif isinstance(op, py.Lt):
        return "l"
    elif isinstance(op, py.LtE):
        return "le"
    else:
        raise UnsupportedNode(op)


def movq(source: x86.arg, destination: x86.arg):
    return x86.Instr("movq", [source, destination])


def negq(destination: x86.arg):
    return x86.Instr("negq", [destination])


def subq(addend: x86.arg, destination: x86.arg):
    return x86.Instr("subq", [addend, destination])


def addq(addend: x86.arg, destination: x86.arg):
    return x86.Instr("addq", [addend, destination])


def xorq(a: x86.arg, b: x86.arg):
    return x86.Instr("xorq", [a, b])


def cmpq(*, rhs: x86.arg, lhs: x86.arg):
    return x86.Instr("cmpq", [rhs, lhs])


def pushq(arg: x86.arg):
    return x86.Instr("pushq", [arg])


def popq(arg: x86.arg):
    return x86.Instr("popq", [arg])


def movzbq_al(destination: x86.arg):
    return x86.Instr("movzbq", [al, destination])


def x86_set(op: py.cmpop):
    return x86.Instr("set" + get_compare_kind(op), [al])


class RemoveJumps(X86Builder):
    def run(self, prog: X86Program) -> X86Program:
        self.blocks = extract_program_body(prog)
        self.links = count_block_links(self.blocks)
        for (label, instrs) in self.blocks.items():
            if self.links[label] == 0 and label not in (start_label, conclusion_label):
                continue
            with self._new_block(label):
                for instr in instrs:
                    self._visit_instr(instr)
        return super()._build()

    def _visit_instr(self, instr: x86.instr):
        if isinstance(instr, x86.Jump) and self.links[instr.label] == 1:
            for i in self.blocks[instr.label]:
                self._emit(i)
        else:
            self._emit(instr)


def count_block_links(body: dict[str, list[x86.instr]]) -> dict[str, int]:
    links = defaultdict(lambda: 0)
    for instrs in body.values():
        for instr in instrs:
            if isinstance(instr, (x86.Jump, x86.JumpIf)):
                links[instr.label] += 1
    return links


main_label = label_name("main")
start_label = label_name("start")
conclusion_label = label_name("conclusion")

callq_read_int = x86.Callq(label_name("read_int"), 0)
callq_print_int = x86.Callq(label_name("print_int"), 1)
retq = x86.Instr("retq", [])

imm1 = x86.Immediate(1)
imm8 = x86.Immediate(8)
al = x86.ByteReg("al")
rax = x86.Reg("rax")
rbx = x86.Reg("rbx")
rcx = x86.Reg("rcx")
rdx = x86.Reg("rdx")
rsp = x86.Reg("rsp")
rbp = x86.Reg("rbp")
rsi = x86.Reg("rsi")
rdi = x86.Reg("rdi")
r8, r9, r10, r11, r12, r13, r14, r15 = (x86.Reg(f"r{i}") for i in range(8, 16))

arg_passing_regs = [rdi, rsi, rcx, rdx, r8, r9]
caller_saved_regs = [rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11]
callee_saved_regs = [rsp, rbp, rbx, r12, r13, r14, r15]

WORD_SIZE = 8  # in bytes


def parse_movq(instr: x86.instr) -> None | tuple[x86.arg, x86.arg]:
    if isinstance(instr, x86.Instr) and instr.instr == "movq":
        assert (
            len(instr.args) == 2
        ), f"Your movq ({instr}) doesn't have exactly 2 arguments"
        [src, dst] = instr.args
        return (src, dst)
    return None


Color = int

reg_to_color: MappingProxyType[x86.Reg, Color] = MappingProxyType(
    {
        r15: -5,
        r11: -4,
        rsp: -2,
        rax: -1,
        rcx: 0,
        rdx: 1,
        rsi: 2,
        rdi: 3,
        r8: 4,
        r9: 5,
        r10: 6,
        rbx: 7,
        r12: 8,
        r13: 9,
        r14: 10,
        rbp: 11,
    }
)
color_to_reg: MappingProxyType[Color, x86.Reg] = MappingProxyType(
    {c: r for (r, c) in reg_to_color.items()}
)
max_reg_color = max(color_to_reg.keys())


class AssignHomes:
    def __init__(self):
        self.n_spilled_vars = 0

    def run(self, prog: X86Program) -> X86Program:
        colors = ColorLocations(prog).run()
        self.used_callees = {
            r
            for c in colors.values()
            if (r := color_to_reg.get(c)) in callee_saved_regs
        }
        self.locations: dict[x86.location, x86.arg] = {
            v: self.__color_to_loc(c) for (v, c) in colors.items()
        }
        return self._visit_program(prog)

    def _visit_program(self, prog: X86Program) -> X86Program:
        body = {}
        for (label, instrs) in extract_program_body(prog).items():
            body[label] = self._visit_instrs(instrs)
        return X86Program(body)

    def _visit_instrs(self, instrs: list[x86.instr]) -> list[x86.instr]:
        return [self._visit_instr(x) for x in instrs]

    def _visit_instr(self, instr: x86.instr) -> x86.instr:
        match instr:
            case x86.Instr(name, args):
                return x86.Instr(name, [self._visit_arg(x) for x in args])
            case _:
                return instr

    def _visit_arg(self, arg: x86.arg) -> x86.arg:
        if isinstance(arg, x86.Variable):
            return self.locations[arg]
        else:
            return arg

    def __color_to_loc(self, c: Color) -> x86.Reg | x86.Deref:
        if (reg := color_to_reg.get(c)) is not None:
            return reg
        index = c - max_reg_color
        self.n_spilled_vars = max(index, self.n_spilled_vars)
        return x86.Deref(rsp.id, (index - 1) * WORD_SIZE)


class ColorLocations:
    def __init__(self, prog: X86Program):
        self.live_memo: dict[str, set[x86.location]] = {}
        self.blocks: dict[str, list[x86.instr]] = extract_program_body(prog)
        self.move_graph: Graph[x86.location] = Graph()
        self.interference_graph: Graph[x86.location] = Graph()

    def run(self) -> dict[location, Color]:
        self._visit_block(start_label)
        return color_graph(self.interference_graph, self.move_graph)

    def _visit_block(self, label: str):
        if label in self.live_memo:
            return
        last_live = set()
        for instr in reversed(self.blocks[label]):
            last_live = self._visit_instr(instr, last_live)
        self.live_memo[label] = last_live

    def _visit_instr(self, i: x86.instr, live_after: set[x86.location]):
        match i:
            case x86.Jump(label):
                return self._get_live_of_block(label)
            case x86.JumpIf(_, label):
                return live_after | self._get_live_of_block(label)
            case _:
                r_set, w_set = get_read_write_sets(i)
                self._update_graphs(i, live_after, r_set, w_set)
                return live_after - w_set | r_set

    def _update_graphs(
        self,
        i: x86.instr,
        live_after: set[x86.location],
        read_set: set[x86.location],
        write_set: set[x86.location],
    ):
        for v in chain(read_set, write_set):
            self.interference_graph.add_vertex(v)
            self.move_graph.add_vertex(v)
        match parse_movq(i):
            case [src, dst]:
                assert isinstance(dst, x86.location), "aaah"
                if isinstance(src, x86.location):
                    self.move_graph.connect(src, dst)
                for loc in live_after:
                    if loc != src and loc != dst:
                        self.interference_graph.connect(dst, loc)
                return
        for w in write_set:
            for loc in live_after:
                if loc != w:
                    self.interference_graph.connect(w, loc)

    def _get_live_of_block(self, label: str) -> set[x86.location]:
        self._visit_block(label)
        return self.live_memo[label]


def get_read_write_sets(i: x86.instr) -> tuple[set[x86.location], set[x86.location]]:
    match i:
        case x86.Instr("movq", [src, dst]):
            rs, ws = [src], [dst]
        case x86.Instr("movzbq", [src, dst]):
            assert src == al, "We don't use other byteregs"
            rs, ws = [rax], [dst]
        case x86.Instr("negq", [dst]):
            rs, ws = [dst], [dst]
        case x86.Instr("subq" | "addq", [arg, dst]):
            rs, ws = [arg, dst], [dst]
        case x86.Callq(name, arity):
            if arity > len(arg_passing_regs):
                raise CallqArityTooBig(name, arity)
            rs = arg_passing_regs[:arity]
            ws = caller_saved_regs
        case x86.Instr("cmpq", [rhs, lhs]):
            rs, ws = [rhs, lhs], []
        case x86.Instr(name) if name.startswith("set"):
            rs, ws = [], [rax]
        case _:
            raise UnsupportedInstr(i)
    return (set(extract_locations(rs)), set(extract_locations(ws)))


def extract_locations(args: Iterable[x86.arg]) -> Iterable[x86.location]:
    for arg in args:
        if isinstance(arg, x86.location):
            yield arg
        elif not isinstance(arg, x86.Immediate):
            raise CompilerError(
                f"discover_live encountered a non-location, non-immediate argument: "
                f": {type(arg)}"
            )


COLOR_LIMIT = 10_000


def color_graph(graph: Graph, move_graph: Graph[x86.location]) -> dict[location, Color]:
    def saturation(v: location) -> set[Color]:
        return {c for x in graph.neighbours(v) if (c := colors.get(x)) is not None}

    def get_move_color(v: location, v_satur: set[Color]) -> Color:
        opts = (colors.get(x, COLOR_LIMIT) for x in move_graph.neighbours(v))
        opts = (c for c in opts if c >= 0 and c not in v_satur)
        return min(opts, default=COLOR_LIMIT)

    def get_free_color(v_satur: set[Color]) -> Color:
        for k in range(0, COLOR_LIMIT):
            if k not in v_satur:
                return k
        raise CompilerError("Too many colors were required")

    def assign_color(v: location, c: Color):
        colors[v] = c
        for u in chain(graph.neighbours(v), move_graph.neighbours(v)):
            if u not in colors:
                queue.increase_key(u)

    def key_f(v: location):
        v_satur = saturation(v)
        return (len(v_satur), get_move_color(v, v_satur), hash(v))

    colors = {v: c for v in graph.vertices() if (c := reg_to_color.get(v)) is not None}
    queue = PriorityQueue(key_to_cmp(key_f))
    for v in graph.vertices():
        if colors.get(v) is None:
            queue.push(v)

    while not queue.empty():
        v = queue.pop()
        satur = saturation(v)
        move_color = get_move_color(v, satur)
        if move_color <= max_reg_color:
            assign_color(v, move_color)
            continue
        free_color = get_free_color(satur)
        if move_color == COLOR_LIMIT or free_color <= max_reg_color:
            assign_color(v, free_color)
        else:
            assign_color(v, move_color)

    return colors


class PatchInstructions(X86Builder):
    def run(self, prog: X86Program) -> X86Program:
        self._visit_program(prog)
        return self._build()

    def _visit_program(self, prog: X86Program):
        for label, instrs in extract_program_body(prog).items():
            with self._new_block(label):
                self._visit_instrs(instrs)

    def _visit_instrs(self, instrs: list[x86.instr]):
        for instr in instrs:
            self._visit_instr(instr)

    def _visit_instr(self, instr: x86.instr):
        match instr:
            case x86.Instr(_, [_, _, _, *_]):
                raise TooManyArgsForInstr(instr)
            case x86.Instr("movq", [src, dst]) if src == dst:
                pass  # such instructions are meaningless and thus skipped
            case x86.Instr("cmpq", [rhs, x86.Immediate() as lhs]):
                self._emit(movq(lhs, rax))
                self._visit_instr(cmpq(rhs=rhs, lhs=rax))
            case x86.Instr(name, [arg1, arg2]) if self.__needs_split(arg1, arg2):
                self._emit(movq(arg1, rax))
                self._emit(x86.Instr(name, [rax, arg2]))
            case _:
                self._emit(instr)

    @staticmethod
    def __needs_split(arg1: x86.arg, arg2: x86.arg) -> bool:
        assert not isinstance(
            arg2, x86.Immediate
        ), "The destination argument shouldn't be an immediate?.."
        return any(
            (
                isinstance(arg1, x86.Deref) and isinstance(arg2, x86.Deref),
                PatchInstructions.__is_big_immediate(arg1),
            )
        )

    @staticmethod
    def __is_big_immediate(arg: x86.arg) -> bool:
        if isinstance(arg, x86.Immediate):
            return not (-(2**16) <= arg.value < 2**16)
        return False


def tracing_res(f):
    def inner(*args, **kwargs):
        res = f(*args, **kwargs)
        print(f.__name__, file=sys.stderr)
        print(res, file=sys.stderr)
        return res

    return inner


class Compiler:
    def __init__(self):
        self.__n_spilled_vars: None | int = None
        self.__used_callees: None | list[x86.Reg] = None

    # @tracing_res
    def partially_evaluate(self, p: py.Module) -> py.Module:
        return PartialEvaluatorIf().run(p)

    # @tracing_res
    def remove_complex_operands(self, p: py.Module, optimize=True) -> py.Module:
        if optimize:
            p = self.partially_evaluate(p)
        return RemoveComplexOperandsIf().run(p)

    # @tracing_res
    def explicate_control(self, p: py.Module) -> CProgram:
        return ExplicateControl().run(p)

    # @tracing_res
    def select_instructions(self, p: CProgram) -> X86Program:
        return SelectInstructionsIf().run(p)

    def remove_jumps(self, p: X86Program) -> X86Program:
        return RemoveJumps().run(p)

    # @tracing_res
    def assign_homes(self, p: X86Program) -> X86Program:
        p = self.remove_jumps(p)
        visitor = AssignHomes()
        result = visitor.run(p)
        self.__n_spilled_vars = visitor.n_spilled_vars + 1
        self.__used_callees = list(visitor.used_callees)
        return result

    # @tracing_res
    def patch_instructions(self, p: X86Program) -> X86Program:
        return PatchInstructions().run(p)

    @tracing_res
    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        body = extract_program_body(p)
        if self.__n_spilled_vars is None or self.__used_callees is None:
            raise MissingPass(self.assign_homes.__qualname__)
        n_used_callees = len(self.__used_callees)
        total_used_space = align16(8 * (n_used_callees + self.__n_spilled_vars))
        vars_used_space = total_used_space - 8 * n_used_callees
        stack_arg = x86.Immediate(vars_used_space)
        reg_prelude: list[x86.instr] = [pushq(x) for x in self.__used_callees]
        reg_conclusion: list[x86.instr] = [popq(x) for x in self.__used_callees[::-1]]
        prelude = reg_prelude + [subq(stack_arg, rsp), x86.Jump(start_label)]
        conclusion = [addq(stack_arg, rsp), *reg_conclusion, retq]
        return X86Program({main_label: prelude, **body, conclusion_label: conclusion})


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def align16(x: int) -> int:
    # weirdly, alignment needs to be 16*n + 8
    q, r = divmod(x, 16)
    if r > 8:
        return 16 * (q + 1) + 8
    else:
        return 16 * q + 8


def extract_program_body(prog: X86Program) -> dict[str, list[x86.instr]]:
    if not isinstance(prog.body, dict):
        raise CompilerError("We now only support blocks")
    return prog.body


def dict_set_fresh(d: dict[K, V], key: K, value: V):
    set_value = d.setdefault(key, value)
    if set_value is not value:
        raise LookupError(f"Key '{key}' is already set in the dict")


def key_to_cmp(key_f):
    return lambda a, b: key_f(a) < key_f(b)


class Graph(Generic[T]):
    def __init__(self, vertices: set[T] = set()):
        self.edges: dict[T, set[T]] = {v: set() for v in vertices}

    def add_vertex(self, v: T) -> None:
        self.edges.setdefault(v, set())

    def connect(self, u: T, v: T):
        self.edges[u].add(v)
        self.edges[v].add(u)

    def neighbours(self, u: T) -> set[T]:
        return self.edges.get(u, set())

    def vertices(self) -> Iterable[T]:
        return self.edges.keys()
