import os

import compiler
import type_check_Lwhile
from interp_Lwhile import InterpLwhile
from interp_x86.eval_x86 import interp_x86
from utils import enable_tracing, run_one_test, run_tests

compiler = compiler.Compiler()

typecheck_Lwhile = type_check_Lwhile.TypeCheckLwhile().type_check

typecheck_dict = {
    "source": typecheck_Lwhile,
    "remove_complex_operands": typecheck_Lwhile,
}
interpLwhile = InterpLwhile().interp
interp_dict = {
    "remove_complex_operands": interpLwhile,
    "select_instructions": interp_x86,
    "assign_homes": interp_x86,
    "patch_instructions": interp_x86,
}


print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")

run_one_test(
    "tests/var/complex-ifs.py", None, compiler, "var", typecheck_dict, interp_dict
)
# run_tests("var", compiler, "var", typecheck_dict, interp_dict)
