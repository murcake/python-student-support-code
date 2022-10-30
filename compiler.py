from __future__ import annotations

import ast as py
import sys
from collections import defaultdict, deque
from collections.abc import Callable, Collection, Iterable
from contextlib import contextmanager
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Generic, Type, TypeVar

import x86_ast as x86
from graph import DirectedAdjList, transpose
from priority_queue import PriorityQueue
from type_check_Ctup import TypeCheckCtup
from type_check_Ltup import TypeCheckLtup
from utils import Allocate as PyAllocate
from utils import Begin as PyBegin
from utils import Collect as PyCollect
from utils import CProgram
from utils import GlobalValue as PyGlobal
from utils import Goto as CGoto
from utils import IntType, TupleType, label_name
from x86_ast import X86Program


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


class ExposeAllocation:
    def __init__(self, name_gen: NameGen):
        self.name_gen = name_gen

    def run(self, p: py.Module) -> py.Module:
        return py.Module(list(self._visit_stmts(p.body)))

    def _visit_stmts(self, stmts: Iterable[py.stmt]) -> Iterable[py.stmt]:
        for stmt in stmts:
            yield self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: py.stmt) -> py.stmt:
        match stmt:
            case py.Assign([lhs_], value_):
                return py.Assign([self._visit_expr(lhs_)], self._visit_expr(value_))
            case py.Expr(expr_):
                return py.Expr(self._visit_expr(expr_))
            case py.If(test_, body_, orelse_):
                test = self._visit_expr(test_)
                body = list(self._visit_stmts(body_))
                orelse = list(self._visit_stmts(orelse_))
                return py.If(test, body, orelse)
            case py.While(test_, body_, orelse_):
                test = self._visit_expr(test_)
                body = list(self._visit_stmts(body_))
                orelse = list(self._visit_stmts(orelse_))
                return py.While(test, body, orelse)
            case _:
                raise UnsupportedNode(stmt)

    def _visit_expr(self, expr: py.expr) -> py.expr:
        match expr:
            case py.Name() | py.Constant():
                return expr
            case py.Call(f_, args_):
                f = self._visit_expr(f_)
                args = [self._visit_expr(x) for x in args_]
                return py.Call(f, args)
            case py.UnaryOp(op, arg_):
                return py.UnaryOp(op, self._visit_expr(arg_))
            case py.BinOp(a_, op, b_):
                return py.BinOp(self._visit_expr(a_), op, self._visit_expr(b_))
            case py.IfExp(test_, body_, orelse_):
                test = self._visit_expr(test_)
                body = self._visit_expr(body_)
                orelse = self._visit_expr(orelse_)
                return py.IfExp(test, body, orelse)
            case py.BoolOp(op, args_):
                return py.BoolOp(op, [self._visit_expr(x) for x in args_])
            case py.Compare(a_, ops, bs_):
                a = self._visit_expr(a_)
                bs = [self._visit_expr(x) for x in bs_]
                return py.Compare(a, ops, bs)
            case py.Tuple(elts_):
                elts = [self._visit_expr(x) for x in elts_]
                return mk_tuple_create(elts, extract_type(expr), self.name_gen)
            case py.Subscript(what_, at_, r):
                return py.Subscript(self._visit_expr(what_), self._visit_expr(at_), r)
            case _:
                raise UnsupportedNode(expr)


def mk_tuple_create(elts: Collection[py.expr], typ: Type, name_gen: NameGen) -> PyBegin:
    length = len(elts)
    n_bytes = 8 + 8 * length
    s = []
    elt_names = []
    for elt in elts:
        elt_name = name_gen.tuple_elt()
        elt_names.append(elt_name)
        s.append(py.Assign([elt_name], elt))
    test_lhs = py.BinOp(py_free_ptr, py.Add(), py.Constant(n_bytes))
    test = py.Compare(test_lhs, [py.GtE()], [py_fromspace_end])
    tuple_name = name_gen.tuple()
    s.append(py.If(test, [PyCollect(n_bytes)], []))
    s.append(py.Assign([tuple_name], PyAllocate(length, typ)))
    for i, elt_name in enumerate(elt_names):
        lhs = py.Subscript(tuple_name, py.Constant(i), py.Store())
        s.append(py.Assign([lhs], elt_name))
    return PyBegin(s, tuple_name)


py_free_ptr = PyGlobal("free_ptr")
py_fromspace_end = PyGlobal("fromspace_end")


class RemoveComplexOperands:
    def __init__(self, name_gen: NameGen):
        self.name_gen = name_gen
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
            case py.Assign([py.Name() as lhs], value_):
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
        name = self.name_gen.tmp()
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

    def _visit_expr(
        self, expr_: py.expr, mk_atomic: bool, locally: bool = False
    ) -> py.expr:
        if not locally:
            return super()._visit_expr(expr_, mk_atomic)
        with self._new_scope():
            expr = super()._visit_expr(expr_, mk_atomic)
            assigns = self._get_tmp_assignments()
            return PyBegin(list(assigns), expr)

    def _visit_expr_meat(self, expr: py.expr, mk_atomic: bool) -> py.expr:
        match expr:
            case py.IfExp(test_, body_, orelse_):
                test = self._visit_expr(test_, mk_atomic=False)
                body = self._visit_expr(body_, mk_atomic=mk_atomic, locally=True)
                orelse = self._visit_expr(orelse_, mk_atomic=mk_atomic, locally=True)
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


class RemoveComplexOperandsWhile(RemoveComplexOperandsIf):
    def _visit_stmt_meat(self, stmt: py.stmt) -> py.stmt:
        match stmt:
            case py.While(test_, body_, orelse_):
                test = self._visit_expr(test_, mk_atomic=False, locally=True)
                body = list(self._visit_stmts(body_))
                orelse = list(self._visit_stmts(orelse_))
                return py.While(test, body, orelse)
            case _:
                return super()._visit_stmt_meat(stmt)


class RemoveComplexOperandsTup(RemoveComplexOperandsWhile):
    def _visit_stmt_meat(self, stmt: py.stmt) -> py.stmt:
        match stmt:
            case py.Assign([py.Subscript() as lhs_], value_):
                lhs = self._visit_expr(lhs_, mk_atomic=False)
                value = self._visit_expr(value_, mk_atomic=False)
                return py.Assign([lhs], value)
            case PyCollect():
                return stmt
            case py.If(py.Subscript() as s):
                new_test = py.Compare(s, [py.Eq()], [pyTrue])
                stmt.test = new_test
                return super()._visit_stmt_meat(stmt)
            case _:
                return super()._visit_stmt_meat(stmt)

    def _visit_expr_meat(self, expr: py.expr, mk_atomic: bool) -> py.expr:
        match expr:
            case py.Subscript(what_, at_, r):
                what = self._visit_expr(what_, mk_atomic=True)
                at = self._visit_expr(at_, mk_atomic=True)
                return py.Subscript(what, at, r)
            case py.Call(py.Name("len"), [arg_]):
                return py.Call(py.Name("len"), [self._visit_expr(arg_, mk_atomic=True)])
            case PyBegin(stmts_, expr_):
                stmts = list(self._visit_stmts(stmts_))
                return PyBegin(stmts, self._visit_expr(expr_, mk_atomic=mk_atomic))
            case PyAllocate() | PyGlobal():
                return expr
            case py.IfExp(py.Subscript() as s):
                new_test = py.Compare(s, [py.Eq()], [pyTrue])
                expr.test = new_test
                return super()._visit_expr_meat(expr, mk_atomic)
            case _:
                return super()._visit_expr_meat(expr, mk_atomic)


class Block:
    name: str
    stmts: list[py.stmt]

    def __init__(self, name: str, stmts: list[py.stmt] | None = None):
        self.name = name
        self.stmts = stmts or []

    def as_jump(self) -> list[py.stmt]:
        match self.stmts:
            case [CGoto()]:
                return self.stmts
        return [CGoto(self.name)]

    def link_to(self, b: Block) -> None:
        if self.stmts and isinstance(self.stmts[-1], CGoto):
            return
        self.stmts.extend(b.as_jump())


class ExplicateControl:
    def __init__(self, name_gen: NameGen):
        self.name_gen = name_gen
        self.current_block: Block = Block(start_label)
        self.blocks: dict[str, list[py.stmt]] = {start_label: self.current_block.stmts}

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
            case py.Expr() | CGoto():
                self._emit(stmt)
            case _:
                raise UnsupportedNode(stmt)

    def _visit_assign(self, lhs: py.Name | py.Subscript, value: py.expr) -> None:
        match value:
            case py.IfExp(test, PyBegin(body_, body_r), PyBegin(orelse_, orelse_r)):
                body_assign = py.Assign([lhs], body_r)
                orelse_assign = py.Assign([lhs], orelse_r)
                self._visit_if(test, body_ + [body_assign], orelse_ + [orelse_assign])
            case py.Constant() | py.Name() | py.BinOp() | py.UnaryOp() | py.Call() | py.Compare():
                self._emit(py.Assign([lhs], value))
            case _:
                raise UnsupportedNode(value)

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
                self._emit(py.If(test, body.as_jump(), orelse.as_jump()))
            case py.UnaryOp(py.Not(), not_test):
                self._visit_if_blocks(not_test, orelse, body)
            case py.IfExp(i_test, i_body, i_orelse):
                body_link = body.as_jump()
                orelse_link = orelse.as_jump()
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
        name = self.name_gen.block_name()
        block = Block(name)
        self.blocks[name] = block.stmts
        return block


class ExplicateControlWhile(ExplicateControl):
    def _visit_stmt(self, stmt: py.stmt) -> None:
        match stmt:
            case py.While(test_, body_):
                head_block = self._mk_block()
                self.current_block.link_to(head_block)
                self._continue_with(head_block)
                body = body_ + head_block.as_jump()
                super()._visit_if(test_, body, [])
            case _:
                super()._visit_stmt(stmt)


class ExplicateControlTup(ExplicateControlWhile):
    def _visit_stmt(self, stmt: py.stmt) -> None:
        match stmt:
            case py.Assign([py.Subscript() as lhs], value_):
                self._visit_assign(lhs, value_)
            case PyCollect():
                self._emit(stmt)
            case _:
                return super()._visit_stmt(stmt)

    def _visit_assign(self, lhs: py.Name | py.Subscript, value: py.expr) -> None:
        match value:
            case PyBegin(stmts, expr):
                self._visit_stmts(stmts)
                self._visit_assign(lhs, expr)
            case PyGlobal() | PyAllocate() | py.Subscript():
                self._emit(py.Assign([lhs], value))
            case _:
                super()._visit_assign(lhs, value)


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
                self._emit(xorq(1, destination))
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


class SelectInstructionsTup(SelectInstructionsIf):
    def __init__(self, name_gen: NameGen):
        super().__init__()
        self.name_gen = name_gen

    def _visit_stmt(self, stmt: py.stmt):
        match stmt:
            case py.Assign([py.Subscript(py.Name(tup), py.Constant(at))], value):
                self._emit(movq(x86.Variable(tup), r11))
                dst = x86.Deref(r11.id, 8 * at + 8)
                return self._visit_assign(dst, value)
            case PyCollect(n_bytes):
                self._emit(movq(r15, rdi))
                self._emit(movq(n_bytes, rsi))
                self._emit(callq_collect)
            case _:
                return super()._visit_stmt(stmt)

    def _visit_assign(self, destination: x86.arg, value: py.expr):
        match value:
            case py.Subscript(py.Name(tup), py.Constant(at)):
                self._emit(movq(x86.Variable(tup), r11))
                src = x86.Deref(r11.id, 8 * at + 8)
                self._emit(movq(src, destination))
            case py.Call(py.Name("len"), [py.Name(tup)]):
                self._emit(movq(x86.Variable(tup), r11))
                self._emit(movq(x86.Deref(r11.id, 0), r11))
                self._emit(sarq(1, r11))
                self._emit(andq(2**6 - 1, r11))
                self._emit(movq(r11, destination))
            case PyAllocate(l, typ):
                self._emit(movq(x86_free_ptr, r11))
                self._emit(addq(8 * l + 8, x86_free_ptr))
                self._emit(movq(mk_tag(typ), x86.Deref(r11.id, 0)))
                self._emit(movq(r11, destination))
            case PyBegin(stmts, expr):
                for stmt in stmts:
                    self._visit_stmt(stmt)
                self._visit_assign(destination, expr)
            case PyGlobal(name):
                self._emit(movq(x86.Global(name), destination))
            case _:
                return super()._visit_assign(destination, value)


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


def mk_tag(typ: Type) -> int:
    if not isinstance(typ, TupleType):
        raise CompilerError(f"Expected a TupleType, got {typ}")
    l = len(typ.types)
    if l > max_tuple_size:
        raise CompilerError(f"The maximum size of tuples allowed is {max_tuple_size}")
    pointer_mask = as_bit_string(map(is_pointer, typ.types))[::-1]
    bits = f"{pointer_mask:>0{max_tuple_size}}{l:06b}0"
    return int(bits, 2)


def is_pointer(typ: Type) -> bool:
    return isinstance(typ, TupleType)


def as_bit_string(bs: Iterable[bool]) -> str:
    return "".join(str(int(x)) for x in bs)


max_tuple_size = 50


def mk_binary_asm_operation(
    name: str,
) -> Callable[[x86.arg | int, x86.arg | int], x86.Instr]:
    def inner(arg: x86.arg | int, destination: x86.arg | int):
        if isinstance(arg, int):
            arg = x86.Immediate(arg)
        if isinstance(destination, int):
            destination = x86.Immediate(destination)
        return x86.Instr(name, [arg, destination])

    return inner


movq = mk_binary_asm_operation("movq")
subq = mk_binary_asm_operation("subq")
addq = mk_binary_asm_operation("addq")
xorq = mk_binary_asm_operation("xorq")
andq = mk_binary_asm_operation("andq")
sarq = mk_binary_asm_operation("sarq")


def negq(destination: x86.arg):
    return x86.Instr("negq", [destination])


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
        self.links[start_label] = -1  # to mark special cases
        self.links[conclusion_label] = -1
        for (label, instrs) in self.blocks.items():
            if self.links[label] == 0:
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

callq_collect = x86.Callq(label_name("collect"), 2)
callq_initialize = x86.Callq(label_name("initialize"), 2)
callq_print_int = x86.Callq(label_name("print_int"), 1)
callq_read_int = x86.Callq(label_name("read_int"), 0)
retq = x86.Instr("retq", [])

x86_free_ptr = x86.Global("free_ptr")
x86_ptrstack_begin = x86.Global("rootstack_begin")
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
        r15: -4,
        r11: -3,
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
    def __init__(self, var_types: dict[str, type]):
        self.var_types: dict[str, Type] = var_types
        self.val_stack: dict[Color, x86_raw_loc] = {}
        self.ptr_stack: dict[Color, x86_raw_loc] = {}
        self.used_callees: set[x86.Reg] = set()
        self.colors: dict[x86.location, Color] = {}

    def run(self, prog: X86Program) -> X86Program:
        self.colors = ColorLocations(prog, self.var_types).run()
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
        if arg in callee_saved_regs:
            self.used_callees.add(arg)
        if isinstance(arg, x86.Variable):
            return self.__get_var_loc(arg)
        else:
            return arg

    def __get_var_loc(self, var: x86.Variable) -> x86_raw_loc:
        c = self.colors[var]
        typ = self.var_types[var.id]
        if is_pointer(typ):
            the_stack = self.ptr_stack
            mk_loc = lambda i: x86.Deref(r15.id, ~i * WORD_SIZE)
        else:
            the_stack = self.val_stack
            mk_loc = lambda i: x86.Deref(rsp.id, i * WORD_SIZE)
        if loc := the_stack.get(c):
            return loc
        loc = mk_loc(len(the_stack))
        the_stack[c] = loc
        return loc


x86_raw_loc = x86.Reg | x86.Deref


class ColorLocations:
    def __init__(self, prog: X86Program, var_types: dict[str, type]):
        self.var_types = var_types
        self.blocks: dict[str, list[x86.instr]] = extract_program_body(prog)
        self.live_befores = get_live_befores(prog)
        self.move_graph: Graph[x86.location] = Graph()
        self.interference_graph: Graph[x86.location] = Graph()

    def run(self) -> dict[x86.location, Color]:
        for label in self.blocks.keys():
            self._visit_block(label)
        return color_graph(self.interference_graph, self.move_graph)

    def _visit_block(self, label: str):
        last_live = set()
        for instr in reversed(self.blocks[label]):
            last_live = self._visit_instr(instr, last_live)

    def _visit_instr(self, i: x86.instr, live_after: set[x86.location]):
        match i:
            case x86.Jump(label):
                return self.live_befores[label]
            case x86.JumpIf(_, label):
                return live_after | self.live_befores[label]
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
        for v in chain(read_set, write_set, live_after):
            self.interference_graph.add_vertex(v)
            self.move_graph.add_vertex(v)
        match parse_movq(i):
            case [src, dst]:
                if not isinstance(dst, x86.location):
                    return
                if isinstance(src, x86.location):
                    self.move_graph.connect(src, dst)
                for loc in live_after:
                    if loc != src and loc != dst:
                        self.interference_graph.connect(dst, loc)
                return
        if isinstance(i, PyCollect):
            for loc in live_after:
                if not self._is_pointer(loc):
                    continue
                for reg in callee_saved_regs:
                    self.interference_graph.connect(reg, loc)
        for w in write_set:
            for loc in live_after:
                if loc != w:
                    self.interference_graph.connect(w, loc)

    def _is_pointer(self, loc: x86.location) -> bool:
        if not isinstance(loc, x86.Variable):
            return False
        typ = self.var_types[loc.id]
        return is_pointer(typ)


def get_live_befores(prog: X86Program) -> dict[str, set[x86.location]]:
    def transfer(node: str, live_after: set[x86.location]) -> set[x86.location]:
        for instr in reversed(blocks[node]):
            if not isinstance(instr, (x86.Jump, x86.JumpIf)):
                r_set, w_set = get_read_write_sets(instr)
                live_after = live_after - w_set | r_set
        return live_after

    blocks = extract_program_body(prog)
    cfg = build_cfg(prog)
    return analyze_dataflow(cfg, transfer, set(), lambda a, b: a | b)


def build_cfg(prog: X86Program) -> DirectedAdjList:
    blocks = extract_program_body(prog)
    g = DirectedAdjList()
    worklist = deque([start_label])
    visited = set()
    while worklist:
        head = worklist.popleft()
        if head in visited:
            continue
        visited.add(head)
        for instr in blocks[head]:
            if isinstance(instr, (x86.Jump, x86.JumpIf)):
                g.add_edge(instr.label, head)
                worklist.append(instr.label)
    return g


def analyze_dataflow(G, transfer, bottom, join) -> dict:
    trans_G = transpose(G)
    mapping = {}
    for v in G.vertices():
        mapping[v] = bottom
    worklist = deque()
    for v in G.vertices():
        worklist.append(v)
    while worklist:
        node = worklist.pop()
        input = reduce(join, [mapping[v] for v in trans_G.adjacent(node)], bottom)
        output = transfer(node, input)
        if output != mapping[node]:
            mapping[node] = output
            for v in G.adjacent(node):
                worklist.append(v)
    return mapping


def get_read_write_sets(i: x86.instr) -> tuple[set[x86.location], set[x86.location]]:
    match i:
        case x86.Instr("movq", [src, dst]):
            rs, ws = [src], [dst]
        case x86.Instr("movzbq", [src, dst]):
            assert src == al, "We don't use other byteregs"
            rs, ws = [rax], [dst]
        case x86.Instr("negq", [dst]):
            rs, ws = [dst], [dst]
        case x86.Instr("subq" | "addq" | "andq" | "sarq", [arg, dst]):
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
    return (set(extract_locations(*rs)), set(extract_locations(*ws)))


def extract_locations(*args: x86.arg) -> Iterable[x86.location]:
    for arg in args:
        if isinstance(arg, x86.location):
            yield arg
        elif isinstance(arg, x86.Deref):
            if arg.reg != r11.id:
                raise CompilerError(f"expected only r11 dereferences, but got: {arg}")
        elif not isinstance(arg, (x86.Immediate, x86.Global)):
            raise CompilerError(
                f"encountered a non-location, non-immediate argument: {type(arg)}"
            )


COLOR_LIMIT = 10_000


def color_graph(
    graph: Graph, move_graph: Graph[x86.location]
) -> dict[x86.location, Color]:
    def saturation(v: x86.location) -> set[Color]:
        return {c for x in graph.neighbours(v) if (c := colors.get(x)) is not None}

    def get_move_color(v: x86.location, v_satur: set[Color]) -> Color:
        opts = (colors.get(x, COLOR_LIMIT) for x in move_graph.neighbours(v))
        opts = (c for c in opts if c >= 0 and c not in v_satur)
        return min(opts, default=COLOR_LIMIT)

    def get_free_color(v_satur: set[Color]) -> Color:
        for k in range(0, COLOR_LIMIT):
            if k not in v_satur:
                return k
        raise CompilerError("Too many colors were required")

    def assign_color(v: x86.location, c: Color):
        colors[v] = c
        for u in chain(graph.neighbours(v), move_graph.neighbours(v)):
            if u not in colors:
                queue.increase_key(u)

    def key_f(v: x86.location):
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
            case x86.Instr("movzbq", [src, dst]) if not isinstance(dst, x86.Reg):
                assert src == al
                self._emit(movzbq_al(rax))
                self._emit(movq(rax, dst))
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
                not (isinstance(arg1, x86.Reg) or isinstance(arg2, x86.Reg)),
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
        self.name_gen = NameGen()
        self.__n_spilled_vals: None | int = None
        self.__n_spilled_ptrs: None | int = None
        self.__used_callees: None | list[x86.Reg] = None
        self.__var_types: None | dict[str, Type] = None

    # @tracing_res
    def expose_allocation(self, p: py.Module) -> py.Module:
        TypeCheckLtup().type_check(p)  # to create 'has_type' fields
        return ExposeAllocation(self.name_gen).run(p)

    # @tracing_res
    def remove_complex_operands(self, p: py.Module) -> py.Module:
        return RemoveComplexOperandsTup(self.name_gen).run(p)

    # @tracing_res
    def explicate_control(self, p: py.Module) -> CProgram:
        return ExplicateControlTup(self.name_gen).run(p)

    # @tracing_res
    def select_instructions(self, p: CProgram) -> X86Program:
        self.__var_types = extract_var_types(p)
        self.__var_types[self.name_gen.the_x86_tmp().id] = IntType()
        return SelectInstructionsTup(self.name_gen).run(p)

    # @tracing_res
    def remove_jumps(self, p: X86Program) -> X86Program:
        return RemoveJumps().run(p)

    # @tracing_res
    def assign_homes(self, p: X86Program) -> X86Program:
        if self.__var_types is None:
            raise MissingPass(self.select_instructions.__qualname__)
        p = self.remove_jumps(p)
        visitor = AssignHomes(self.__var_types)
        result = visitor.run(p)
        self.__n_spilled_vals = len(visitor.val_stack)
        self.__n_spilled_ptrs = len(visitor.ptr_stack)
        self.__used_callees = list(visitor.used_callees)
        return result

    # @tracing_res
    def patch_instructions(self, p: X86Program) -> X86Program:
        return PatchInstructions().run(p)

    # @tracing_res
    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        body = extract_program_body(p)
        if (
            self.__n_spilled_vals is None
            or self.__n_spilled_ptrs is None
            or self.__used_callees is None
        ):
            raise MissingPass(self.assign_homes.__qualname__)
        n_used_callees = len(self.__used_callees)
        total_used_stack = align_stack(8 * (n_used_callees + self.__n_spilled_vals))
        vals_used_stack = total_used_stack - 8 * n_used_callees

        prelude: list[x86.instr] = [
            *(pushq(x) for x in self.__used_callees),
            subq(vals_used_stack, rsp),
            movq(ptr_stack_size, rdi),
            movq(heap_size, rsi),
            callq_initialize,
            movq(x86_ptrstack_begin, r15),
            *(movq(0, x86.Deref(r15.id, 8 * i)) for i in range(self.__n_spilled_ptrs)),
            addq(self.__n_spilled_ptrs, r15),
            x86.Jump(start_label),
        ]

        conclusion: list[x86.instr] = [
            subq(self.__n_spilled_ptrs, r15),
            addq(vals_used_stack, rsp),
            *(popq(x) for x in reversed(self.__used_callees)),
            retq,
        ]
        return X86Program({main_label: prelude, **body, conclusion_label: conclusion})


ptr_stack_size = 64 * 256
heap_size = 64 * 256

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def align_stack(x: int) -> int:
    # alignment needs to be 16*n + 8, because address is already on the stack
    q, r = divmod(x, 16)
    if r > 8:
        return 16 * (q + 1) + 8
    else:
        return 16 * q + 8


def extract_program_body(prog: X86Program) -> dict[str, list[x86.instr]]:
    if not isinstance(prog.body, dict):
        raise CompilerError("We now only support blocks")
    return prog.body


def extract_type(node: py.AST) -> Type:
    if typ := getattr(node, "has_type"):
        return typ
    raise MissingPass("TypeCheckLtup")


def extract_var_types(p: CProgram) -> dict[str, Type]:
    TypeCheckCtup().type_check(p)  # to create 'var_types' fields
    if (typ := getattr(p, "var_types")) is not None:
        return typ
    raise AssertionError("TypeCheckCtup didn't create 'var_types'?")


def dict_set_fresh(d: dict[K, V], key: K, value: V):
    set_value = d.setdefault(key, value)
    if set_value is not value:
        raise LookupError(f"Key '{key}' is already set in the dict")


def key_to_cmp(key_f):
    return lambda a, b: key_f(a) < key_f(b)


class Graph(Generic[T]):
    def __init__(self, vertices: set[T] | None = None):
        self.edges: dict[T, set[T]] = {v: set() for v in vertices or set()}

    def add_vertex(self, v: T) -> None:
        self.edges.setdefault(v, set())

    def connect(self, u: T, v: T):
        self.edges[u].add(v)
        self.edges[v].add(u)

    def neighbours(self, u: T) -> set[T]:
        return self.edges.get(u, set())

    def vertices(self) -> Iterable[T]:
        return self.edges.keys()


class NameGen:
    def __init__(self):
        self._n_tuple_elts = 0
        self._n_blocks = 0
        self._n_tmps = 0
        self._n_tuples = 0

    def tuple_elt(self) -> py.Name:
        self._n_tuple_elts += 1
        return py.Name(f"$tuple_elt#{self._n_tuple_elts}")

    def block_name(self) -> str:
        self._n_blocks += 1
        return label_name(f"block_{self._n_blocks}")

    def tuple(self) -> py.Name:
        self._n_tuples += 1
        return py.Name(f"$tuple#{self._n_tuples}")

    def tmp(self) -> py.Name:
        self._n_tmps += 1
        return py.Name(f"$tmp#{self._n_tmps}")

    def the_x86_tmp(self) -> x86.Variable:
        return x86.Variable("$tmp")
