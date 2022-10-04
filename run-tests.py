import compiler
import type_check_Lif
from interp_Lif import InterpLif
from interp_x86.eval_x86 import interp_x86
from utils import enable_tracing, run_one_test, run_tests

compiler = compiler.Compiler()

typecheck_Lif = type_check_Lif.TypeCheckLif().type_check

typecheck_dict = {
    "source": typecheck_Lif,
    "remove_complex_operands": typecheck_Lif,
}
interpLif = InterpLif().interp
interp_dict = {
    "remove_complex_operands": interpLif,
    "select_instructions": interp_x86,
    "assign_homes": interp_x86,
    "patch_instructions": interp_x86,
}


# run_one_test(
#     "tests/var/complex-ifs.py", None, compiler, "var", typecheck_dict, interp_dict
# )
run_tests("var", compiler, "var", typecheck_dict, interp_dict)
