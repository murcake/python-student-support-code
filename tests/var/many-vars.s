	.globl main
main:
    subq $8, %rsp
    movq $1, %rax
    movq %rax, %rcx
    movq $42, %rax
    movq %rax, %rcx
    movq $8, %rax
    movq %rax, %rcx
    movq $8, %rax
    movq %rax, %rcx
    movq $50, %rax
    movq %rax, %rcx
    movq $42, %rax
    movq %rax, %rdi
    callq print_int
    addq $8, %rsp
    retq 

