import ast as py
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TypeVar

import x86_ast as x86
from utils import label_name
from x86_ast import X86Program


class UnsupportedNode(Exception):
    def __init__(self, node: py.AST):
        msg = f"The following node is not supported by the language: {repr(node)}"
        super().__init__(msg)


class NonAtomicExpr(Exception):
    def __init__(self, e: py.expr):
        msg = f"The following expression is not atomic: {e}"
        super().__init__(msg)


class TooManyArgsForInstr(Exception):
    def __init__(self, i: x86.Instr):
        msg = f"The following instruction has too many (> 2) arguments: {i}"
        super().__init__(msg)


class MissingPass(Exception):
    def __init__(self, pass_name: str):
        msg = f"Complete the following pass first: f{pass_name}"
        super().__init__(msg)


class PartialEvaluatorLVar:
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


class RemoveComplexOperands:
    def __init__(self):
        self.tmp_cnt: int = 0
        self.tmp_bindings: list[tuple[py.Name, py.expr]] = []

    def run(self, p: py.Module) -> py.Module:
        return self._visit_module(p)

    def _visit_module(self, p: py.Module) -> py.Module:
        body = []
        for stmt in p.body:
            body.extend(self._visit_stmt(stmt))
        return py.Module(body)

    def _visit_stmt(self, stmt: py.stmt) -> Iterable[py.stmt]:
        with self._new_stmt():
            match stmt:
                case py.Assign([lhs], value_):
                    main = py.Assign([lhs], self._visit_expr(value_, is_rhs=True))
                case py.Expr(py.Call(py.Name("print"), args_)):
                    args = [self._visit_expr(arg, is_rhs=False) for arg in args_]
                    main = py.Expr(py.Call(py.Name("print"), args))
                case py.Expr(expr):
                    main = py.Expr(self._visit_expr(expr, is_rhs=True))
                case _:
                    raise UnsupportedNode(stmt)
            yield from (py.Assign([n], v) for (n, v) in self.tmp_bindings)
            yield main

    def _visit_expr(self, expr: py.expr, is_rhs: bool) -> py.expr:
        match expr:
            case py.Name() | py.Constant():
                return expr
            case py.Call(py.Name("input_int")):
                main = expr
            case py.UnaryOp(py.USub(), arg):
                main = py.UnaryOp(py.USub(), self._visit_expr(arg, is_rhs=False))
            case py.BinOp(arg1_, (py.Add() | py.Sub()) as op, arg2_):
                arg1 = self._visit_expr(arg1_, is_rhs=False)
                arg2 = self._visit_expr(arg2_, is_rhs=False)
                main = py.BinOp(arg1, op, arg2)
            case _:
                raise UnsupportedNode(expr)
        if not is_rhs:
            main = self._bind_to_tmp(main)
        return main

    def _bind_to_tmp(self, value: py.expr) -> py.Name:
        name = py.Name(f"$tmp_{self.tmp_cnt}")
        self.tmp_cnt += 1
        self.tmp_bindings.append((name, value))
        return name

    @contextmanager
    def _new_stmt(self):
        old_tmp_bindings = self.tmp_bindings
        self.tmp_bindings = []
        try:
            yield None
        finally:
            self.tmp_bindings = old_tmp_bindings


# inspired by a writer monad
class X86Builder:
    def __init__(self):
        self.__current_function: list[x86.instr] = []
        self.__x86_body: dict[str, list[x86.instr]] = {"main": self.__current_function}

    def _emit(self, i: x86.instr):
        self.__current_function.append(i)

    def _build(self) -> X86Program:
        if len(self.__x86_body) == 1:
            return X86Program(self.__current_function)
        return X86Program(self.__x86_body)

    @contextmanager
    def _new_function(self, name: str):
        old_current_function = self.__current_function
        self.__current_function = []
        dict_set_fresh(self.__x86_body, name, self.__current_function)
        try:
            yield None
        finally:
            self.__current_function = old_current_function


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


def try_take_atomic_arg(e: py.expr) -> None | x86.arg:
    match e:
        case py.Name(n):
            return x86.Variable(n)
        case py.Constant(v):
            return x86.Immediate(v)
        case _:
            return None


def take_atomic_arg(e: py.expr) -> x86.arg:
    if res := try_take_atomic_arg(e):
        return res
    else:
        raise NonAtomicExpr(e)


def movq(source: x86.arg, destination: x86.arg):
    return x86.Instr("movq", [source, destination])


def negq(destination: x86.arg):
    return x86.Instr("negq", [destination])


def subq(addend: x86.arg, destination: x86.arg):
    return x86.Instr("subq", [addend, destination])


def addq(addend: x86.arg, destination: x86.arg):
    return x86.Instr("addq", [addend, destination])


def pushq(arg: x86.arg):
    return x86.Instr("pushq", [arg])


def popq(arg: x86.arg):
    return x86.Instr("popq", [arg])


callq_read_int = x86.Callq(label_name("read_int"), 0)
callq_print_int = x86.Callq(label_name("print_int"), 1)
retq = x86.Instr("retq", [])
rax = x86.Reg("rax")
rdi = x86.Reg("rdi")
rbp = x86.Reg("rbp")
rsp = x86.Reg("rsp")

WORD_SIZE = 8  # in bytes


class AssignHomes:
    def __init__(self):
        self.indices: dict[str, int] = {}

    def run(self, prog: X86Program) -> X86Program:
        return self._visit_program(prog)

    def get_used_stack_space(self) -> int:
        return len(self.indices) * WORD_SIZE

    def _visit_program(self, prog: X86Program) -> X86Program:
        assert isinstance(prog.body, list), "Multiple functions aren't supported yet"
        return X86Program(self._visit_instrs(prog.body))

    def _visit_instrs(self, instrs: list[x86.instr]) -> list[x86.instr]:
        return [self._visit_instr(x) for x in instrs]

    def _visit_instr(self, instr: x86.instr) -> x86.instr:
        match instr:
            case x86.Instr(name, args):
                return x86.Instr(name, [self._visit_arg(x) for x in args])
            case _:
                return instr

    def _visit_arg(self, arg: x86.arg) -> x86.arg:
        match arg:
            case x86.Variable(name):
                index = self.indices.setdefault(name, len(self.indices))
                return x86.Deref(rbp.id, -(index + 1) * WORD_SIZE)
            case _:
                return arg


class PatchInstructions(X86Builder):
    def run(self, prog: X86Program) -> X86Program:
        self._visit_program(prog)
        return self._build()

    def _visit_program(self, prog: X86Program):
        if isinstance(prog.body, dict):
            for label, instrs in prog.body.items():
                with self._new_function(label):
                    self._visit_instrs(instrs)
        else:
            self._visit_instrs(prog.body)

    def _visit_instrs(self, instrs: list[x86.instr]):
        for instr in instrs:
            self._visit_instr(instr)

    def _visit_instr(self, instr: x86.instr):
        match instr:
            case x86.Instr(_, [_, _, _, *_]):
                raise TooManyArgsForInstr(instr)
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
            return not (2**16 <= arg.value < 2**16)
        return False


class Compiler:
    def __init__(self):
        self.__used_stack_space: None | int = None

    def remove_complex_operands(self, p: py.Module) -> py.Module:
        p = PartialEvaluatorLVar().run(p)
        return RemoveComplexOperands().run(p)

    def select_instructions(self, p: py.Module) -> X86Program:
        return SelectInstructions().run(p)

    def assign_homes(self, p: X86Program) -> X86Program:
        visitor = AssignHomes()
        result = visitor.run(p)
        self.__used_stack_space = visitor.get_used_stack_space()
        return result

    def patch_instructions(self, p: X86Program) -> X86Program:
        return PatchInstructions().run(p)

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        assert isinstance(p.body, list), "Multiple functions aren't supported yet"
        if self.__used_stack_space is None:
            raise MissingPass(self.assign_homes.__qualname__)
        if self.__used_stack_space == 0:
            prelude, conclusion = [], [retq]
        else:
            self.__used_stack_space += 8  # for rsp
            if self.__used_stack_space % 16 != 0:  # align to 16
                assert self.__used_stack_space % 16 == 8
                self.__used_stack_space += 8
            stack_arg = x86.Immediate(self.__used_stack_space)
            prelude = [pushq(rbp), movq(rsp, rbp), subq(stack_arg, rsp)]
            conclusion = [addq(stack_arg, rsp), popq(rbp), retq]
        return X86Program(prelude + p.body + conclusion)


K = TypeVar("K")
V = TypeVar("V")


def dict_set_fresh(d: dict[K, V], key: K, value: V):
    set_value = d.setdefault(key, value)
    if set_value is not value:
        raise LookupError(f"Key '{key}' is already set in the dict")
