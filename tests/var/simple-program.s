	.globl main
main:
    pushq %rbx
    subq $16, %rsp
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, %rdi
    addq %rbx, %rdi
    callq print_int
    addq $16, %rsp
    popq %rbx
    retq 

