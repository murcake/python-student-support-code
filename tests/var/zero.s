	.globl main
main:
    subq $8, %rsp
    movq $0, %rax
    movq %rax, %rdi
    callq print_int
    addq $8, %rsp
    retq 

