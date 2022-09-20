from __future__ import annotations

import ast as py
from collections.abc import Iterable
from contextlib import contextmanager
from itertools import chain
from types import MappingProxyType
from typing import Generic, TypeVar

import x86_ast as x86
from priority_queue import PriorityQueue
from utils import label_name
from x86_ast import X86Program, location


class CompilerError(Exception):
    pass


class UnsupportedNode(CompilerError):
    def __init__(self, node: py.AST):
        msg = f"The following node is not supported by the language: {repr(node)}"
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


class MultipleFunctionsNotSupported(CompilerError):
    def __init__(self):
        super().__init__("Multiple functions aren't supported yet")


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

imm8 = x86.Immediate(8)
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
        interference_graph = build_interference(prog)
        move_graph = build_move_graph(prog)
        colors = color_graph(interference_graph, move_graph)
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
        return X86Program(self._visit_instrs(extract_program_body(prog)))

    def _visit_instrs(self, instrs: list[x86.instr]) -> list[x86.instr]:
        return [self._visit_instr(x) for x in instrs]

    def _visit_instr(self, instr: x86.instr) -> x86.instr:
        match instr:
            case x86.Instr(name, args):
                return x86.Instr(name, [self._visit_arg(x) for x in args])
            case _:
                return instr

    def _visit_arg(self, arg: x86.arg) -> x86.arg:
        if isinstance(arg, x86.location):
            return self.locations[arg]
        else:
            return arg

    def __color_to_loc(self, c: Color) -> x86.Reg | x86.Deref:
        if (reg := color_to_reg.get(c)) is not None:
            return reg
        index = c - max_reg_color
        self.n_spilled_vars = max(index, self.n_spilled_vars)
        return x86.Deref(rsp.id, (index - 1) * WORD_SIZE)


def uncover_live(prog: X86Program) -> list[set[x86.location]]:
    body = extract_program_body(prog)
    res: list[set[x86.location]] = [set()]
    for instr in reversed(body):
        last_set = res[-1]
        r_set, w_set = get_read_write_sets(instr)
        res.append(last_set - w_set | r_set)
    res.pop()  # remove the set before all instructions
    res.reverse()
    return res


# @lru_cache
def get_read_write_sets(i: x86.instr) -> tuple[set[x86.location], set[x86.location]]:
    match i:
        case x86.Instr("movq", [src, dst]):
            rs, ws = [src], [dst]
        case x86.Instr("negq", [dst]):
            rs, ws = [dst], [dst]
        case x86.Instr("subq" | "addq", [arg, dst]):
            rs, ws = [arg, dst], [dst]
        case x86.Callq(name, arity):
            if arity > len(arg_passing_regs):
                raise CallqArityTooBig(name, arity)
            rs = arg_passing_regs[:arity]
            ws = caller_saved_regs
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


def get_access_set(i: x86.instr) -> set[x86.location]:
    rs, ws = get_read_write_sets(i)
    return rs | ws


def get_all_program_locations(body: list[x86.instr]) -> set[x86.location]:
    return set().union(*(get_access_set(x) for x in body))


def build_interference(prog: X86Program) -> Graph[x86.location]:
    body = extract_program_body(prog)
    live_sets = uncover_live(prog)
    all_locations = get_all_program_locations(body)
    assert len(body) == len(live_sets), "There must be a lives set for each instruction"
    graph = Graph(all_locations)
    for instr, live_set in zip(body, live_sets):
        match parse_movq(instr):
            case [src, dst]:
                assert isinstance(
                    dst, x86.location
                ), "shouldn't be otherwise at this stage"
                for loc in live_set:
                    if loc != src and loc != dst:
                        graph.connect(dst, loc)
                continue
        _, write_set = get_read_write_sets(instr)
        for w in write_set:
            for loc in live_set:
                if loc != w:
                    graph.connect(w, loc)
    return graph


def build_move_graph(prog: X86Program) -> Graph[x86.location]:
    body = extract_program_body(prog)
    all_locations = get_all_program_locations(body)
    graph = Graph(all_locations)
    for instr in body:
        match parse_movq(instr):
            case [x86.location() as src, dst]:
                assert isinstance(dst, x86.location), "aaah"
                graph.connect(src, dst)
    return graph


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
            case x86.Instr("movq", [src, dst]) if src == dst:
                pass  # such instructions are meaningless and thus skipped
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
        self.__n_spilled_vars: None | int = None
        self.__used_callees: None | list[x86.Reg] = None

    def remove_complex_operands(self, p: py.Module, optimize=True) -> py.Module:
        if optimize:
            p = PartialEvaluatorLVar().run(p)
        return RemoveComplexOperands().run(p)

    def select_instructions(self, p: py.Module) -> X86Program:
        return SelectInstructions().run(p)

    def assign_homes(self, p: X86Program) -> X86Program:
        visitor = AssignHomes()
        result = visitor.run(p)
        self.__n_spilled_vars = visitor.n_spilled_vars + 1
        self.__used_callees = list(visitor.used_callees)
        return result

    def patch_instructions(self, p: X86Program) -> X86Program:
        return PatchInstructions().run(p)

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        body = extract_program_body(p)
        if self.__n_spilled_vars is None or self.__used_callees is None:
            raise MissingPass(self.assign_homes.__qualname__)
        n_used_callees = len(self.__used_callees)
        total_used_space = align16(8 * (n_used_callees + self.__n_spilled_vars))
        vars_used_space = total_used_space - 8 * n_used_callees
        stack_arg = x86.Immediate(vars_used_space)
        reg_prelude = [pushq(x) for x in self.__used_callees]
        reg_conclusion = [popq(x) for x in reversed(self.__used_callees)]
        prelude = reg_prelude + [subq(stack_arg, rsp)]
        conclusion = [addq(stack_arg, rsp), *reg_conclusion, retq]
        return X86Program(prelude + body + conclusion)


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


def extract_program_body(prog: X86Program) -> list[x86.instr]:
    if not isinstance(prog.body, list):
        raise MultipleFunctionsNotSupported()
    return prog.body


def dict_set_fresh(d: dict[K, V], key: K, value: V):
    set_value = d.setdefault(key, value)
    if set_value is not value:
        raise LookupError(f"Key '{key}' is already set in the dict")


def key_to_cmp(key_f):
    return lambda a, b: key_f(a) < key_f(b)


class Graph(Generic[T]):
    def __init__(self, vertices: set[T]):
        self.edges: dict[T, set[T]] = {v: set() for v in vertices}

    def connect(self, u: T, v: T):
        self.edges[u].add(v)
        self.edges[v].add(u)

    def neighbours(self, u: T) -> set[T]:
        return self.edges.get(u, set())

    def vertices(self) -> Iterable[T]:
        return self.edges.keys()
