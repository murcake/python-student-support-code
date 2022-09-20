	.globl main
main:
    pushq %rbx
    subq $16, %rsp
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, %rcx
    callq read_int
    movq %rax, %rcx
    callq read_int
    movq %rax, %rcx
    addq %rbx, %rcx
    movq $2, %rax
    addq %rax, %rcx
    movq %rcx, %rdi
    callq print_int
    addq $16, %rsp
    popq %rbx
    retq 

