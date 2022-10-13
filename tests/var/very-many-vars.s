	.globl main
	.align 16
main:
    pushq %r14
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    subq $96, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, 0(%rsp)
    callq read_int
    movq %rax, 16(%rsp)
    callq read_int
    movq %rax, 48(%rsp)
    callq read_int
    movq %rax, %r14
    callq read_int
    movq %rax, 8(%rsp)
    callq read_int
    movq %rax, 40(%rsp)
    callq read_int
    movq %rax, 32(%rsp)
    callq read_int
    movq %rax, %r13
    callq read_int
    movq %rax, 24(%rsp)
    callq read_int
    movq %rax, %rbp
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, %r12
    callq read_int
    movq %rax, 72(%rsp)
    callq read_int
    movq %rax, 64(%rsp)
    callq read_int
    movq %rax, 56(%rsp)
    callq read_int
    movq %rax, %rdi
    movq 16(%rsp), %rcx
    negq %rcx
    movq 0(%rsp), %rdx
    addq %rcx, %rdx
    negq %r14
    movq 48(%rsp), %rcx
    addq %r14, %rcx
    addq %rcx, %rdx
    movq 40(%rsp), %rsi
    negq %rsi
    movq 8(%rsp), %rcx
    addq %rsi, %rcx
    addq %rcx, %rdx
    negq %r13
    movq 32(%rsp), %rsi
    addq %r13, %rsi
    movq %rdx, %rcx
    addq %rsi, %rcx
    negq %rbp
    movq 24(%rsp), %rdx
    addq %rbp, %rdx
    addq %rdx, %rcx
    negq %r12
    addq %r12, %rbx
    movq %rcx, %rdx
    addq %rbx, %rdx
    movq 64(%rsp), %rcx
    negq %rcx
    movq 72(%rsp), %rsi
    addq %rcx, %rsi
    movq %rdx, %rcx
    addq %rsi, %rcx
    movq %rdi, %rdx
    negq %rdx
    movq 56(%rsp), %rsi
    addq %rdx, %rsi
    addq %rsi, %rcx
    movq %rcx, %rdi
    addq $42, %rdi
    callq print_int
    movq $0, %rax

	.align 16
conclusion:
    addq $96, %rsp
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    popq %r14
    retq 


