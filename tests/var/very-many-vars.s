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
    movq %rax, 8(%rsp)
    callq read_int
    movq %rax, 56(%rsp)
    callq read_int
    movq %rax, 16(%rsp)
    callq read_int
    movq %rax, %rbp
    callq read_int
    movq %rax, 24(%rsp)
    callq read_int
    movq %rax, 64(%rsp)
    callq read_int
    movq %rax, 32(%rsp)
    callq read_int
    movq %rax, 72(%rsp)
    callq read_int
    movq %rax, 40(%rsp)
    callq read_int
    movq %rax, 0(%rsp)
    callq read_int
    movq %rax, %r12
    callq read_int
    movq %rax, 48(%rsp)
    callq read_int
    movq %rax, %r13
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, %r14
    callq read_int
    movq %rax, %rdx
    movq 56(%rsp), %rcx
    negq %rcx
    movq 8(%rsp), %rsi
    addq %rcx, %rsi
    movq %rbp, %rcx
    negq %rcx
    movq 16(%rsp), %rdi
    addq %rcx, %rdi
    addq %rdi, %rsi
    movq 64(%rsp), %rcx
    negq %rcx
    movq 24(%rsp), %rdi
    addq %rcx, %rdi
    addq %rdi, %rsi
    movq 72(%rsp), %rdi
    negq %rdi
    movq 32(%rsp), %rcx
    addq %rdi, %rcx
    addq %rcx, %rsi
    movq 0(%rsp), %rdi
    negq %rdi
    movq 40(%rsp), %rcx
    addq %rdi, %rcx
    addq %rcx, %rsi
    movq 48(%rsp), %rcx
    negq %rcx
    addq %rcx, %r12
    movq %rsi, %rcx
    addq %r12, %rcx
    negq %rbx
    addq %rbx, %r13
    movq %rcx, %rsi
    addq %r13, %rsi
    movq %rdx, %rcx
    negq %rcx
    movq %r14, %rdx
    addq %rcx, %rdx
    movq %rsi, %rcx
    addq %rdx, %rcx
    addq $42, %rcx
    movq %rcx, %rdi
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


