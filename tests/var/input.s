	.globl main
main:
    subq $8, %rsp
    callq read_int
    movq %rax, %rdi
    callq print_int
    addq $8, %rsp
    retq 

