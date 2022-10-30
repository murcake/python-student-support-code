import os

import compiler
import type_check_Ltup
from interp_Ctup import InterpCtup
from interp_Ltup import InterpLtup
from interp_x86.eval_x86 import interp_x86
from utils import enable_tracing, run_one_test, run_tests

compiler = compiler.Compiler()

typecheck_Ltup = type_check_Ltup.TypeCheckLtup().type_check

typecheck_dict = {
    "source": typecheck_Ltup,
    "remove_complex_operands": typecheck_Ltup,
}
interpLtup = InterpLtup().interp
interpCtup = InterpCtup().interp
interp_dict = {
    "remove_complex_operands": interpLtup,
    "explicate_control": interpCtup,
    # "select_instructions": interp_x86,
    # "assign_homes": interp_x86,
    # "patch_instructions": interp_x86,
}


print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")

the_test = "tests/var/tuple-access.py"
# run_one_test(the_test, None, compiler, "var", typecheck_dict, interp_dict)
run_tests("var", compiler, "var", typecheck_dict, interp_dict)
