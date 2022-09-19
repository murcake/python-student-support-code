	.globl main
main:
    movq $42, %rax
    movq %rax, %rdi
    callq print_int
    retq 

