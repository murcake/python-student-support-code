	.globl main
	.align 16
main:
    pushq %r12
    pushq %r14
    pushq %rbp
    pushq %r13
    pushq %rbx
    subq $96, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, %r14
    callq read_int
    movq %rax, %rbp
    callq read_int
    movq %rax, 32(%rsp)
    callq read_int
    movq %rax, 56(%rsp)
    callq read_int
    movq %rax, 16(%rsp)
    callq read_int
    movq %rax, 24(%rsp)
    callq read_int
    movq %rax, 48(%rsp)
    callq read_int
    movq %rax, 40(%rsp)
    callq read_int
    movq %rax, 72(%rsp)
    callq read_int
    movq %rax, 8(%rsp)
    callq read_int
    movq %rax, 64(%rsp)
    callq read_int
    movq %rax, 0(%rsp)
    callq read_int
    movq %rax, %r13
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, %r12
    callq read_int
    movq %rax, %rcx
    negq %rbp
    movq %r14, %rsi
    addq %rbp, %rsi
    movq 56(%rsp), %rdi
    negq %rdi
    movq 32(%rsp), %rdx
    addq %rdi, %rdx
    addq %rdx, %rsi
    movq 24(%rsp), %rdi
    negq %rdi
    movq 16(%rsp), %rdx
    addq %rdi, %rdx
    addq %rdx, %rsi
    movq 40(%rsp), %rdi
    negq %rdi
    movq 48(%rsp), %rdx
    addq %rdi, %rdx
    addq %rdx, %rsi
    movq 8(%rsp), %rdx
    negq %rdx
    movq 72(%rsp), %rdi
    addq %rdx, %rdi
    addq %rdi, %rsi
    movq 0(%rsp), %rdx
    negq %rdx
    movq 64(%rsp), %rdi
    addq %rdx, %rdi
    movq %rsi, %r8
    addq %rdi, %r8
    movq %rbx, %rsi
    negq %rsi
    movq %r13, %rdx
    addq %rsi, %rdx
    movq %r8, %rsi
    addq %rdx, %rsi
    negq %rcx
    movq %r12, %rdx
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
    popq %rbx
    popq %r13
    popq %rbp
    popq %r14
    popq %r12
    retq 


