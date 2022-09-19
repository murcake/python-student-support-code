	.globl main
main:
    movq $0, %rax
    movq %rax, %rdi
    callq print_int
    retq 

