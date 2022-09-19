	.globl main
main:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    callq read_int
    movq %rax, -8(%rbp)
    callq read_int
    movq %rax, -16(%rbp)
    callq read_int
    movq %rax, -24(%rbp)
    callq read_int
    movq %rax, -32(%rbp)
    movq -32(%rbp), %rax
    movq %rax, -40(%rbp)
    movq -8(%rbp), %rax
    addq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    movq %rax, -48(%rbp)
    movq -8(%rbp), %rax
    addq %rax, -48(%rbp)
    movq -8(%rbp), %rax
    movq %rax, -56(%rbp)
    negq -56(%rbp)
    movq -48(%rbp), %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    addq %rax, -64(%rbp)
    movq -64(%rbp), %rax
    movq %rax, -72(%rbp)
    movq $2, %rax
    addq %rax, -72(%rbp)
    movq -72(%rbp), %rdi
    callq print_int
    addq $80, %rsp
    popq %rbp
    retq 

