	.globl main
main:
    pushq %r14
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    subq $96, %rsp
    callq read_int
    movq %rax, 32(%rsp)
    callq read_int
    movq %rax, %r14
    callq read_int
    movq %rax, 72(%rsp)
    callq read_int
    movq %rax, 48(%rsp)
    callq read_int
    movq %rax, 0(%rsp)
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, 40(%rsp)
    callq read_int
    movq %rax, %r13
    callq read_int
    movq %rax, 24(%rsp)
    callq read_int
    movq %rax, 16(%rsp)
    callq read_int
    movq %rax, %r12
    callq read_int
    movq %rax, 56(%rsp)
    callq read_int
    movq %rax, 8(%rsp)
    callq read_int
    movq %rax, %rbp
    callq read_int
    movq %rax, 64(%rsp)
    callq read_int
    movq %rax, %rdi
    negq %r14
    movq 32(%rsp), %rcx
    addq %r14, %rcx
    movq 48(%rsp), %rdx
    negq %rdx
    movq 72(%rsp), %rsi
    addq %rdx, %rsi
    movq %rcx, %rdx
    addq %rsi, %rdx
    negq %rbx
    movq 0(%rsp), %rcx
    addq %rbx, %rcx
    addq %rcx, %rdx
    movq %r13, %rcx
    negq %rcx
    movq 40(%rsp), %rsi
    addq %rcx, %rsi
    movq %rdx, %rcx
    addq %rsi, %rcx
    movq 16(%rsp), %rsi
    negq %rsi
    movq 24(%rsp), %rdx
    addq %rsi, %rdx
    movq %rcx, %rsi
    addq %rdx, %rsi
    movq 56(%rsp), %rcx
    negq %rcx
    addq %rcx, %r12
    movq %rsi, %rdx
    addq %r12, %rdx
    negq %rbp
    movq 8(%rsp), %rcx
    addq %rbp, %rcx
    addq %rcx, %rdx
    negq %rdi
    movq 64(%rsp), %rcx
    addq %rdi, %rcx
    addq %rcx, %rdx
    movq $42, %rax
    addq %rax, %rdx
    movq %rdx, %rdi
    callq print_int
    addq $96, %rsp
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    popq %r14
    retq 

