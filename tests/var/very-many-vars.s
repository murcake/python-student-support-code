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
    movq %rax, 56(%rsp)
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, 40(%rsp)
    callq read_int
    movq %rax, 32(%rsp)
    callq read_int
    movq %rax, %rbp
    callq read_int
    movq %rax, 8(%rsp)
    callq read_int
    movq %rax, 0(%rsp)
    callq read_int
    movq %rax, 16(%rsp)
    callq read_int
    movq %rax, 48(%rsp)
    callq read_int
    movq %rax, %r14
    callq read_int
    movq %rax, %r13
    callq read_int
    movq %rax, 24(%rsp)
    callq read_int
    movq %rax, 64(%rsp)
    callq read_int
    movq %rax, %r12
    callq read_int
    movq %rax, 72(%rsp)
    callq read_int
    movq %rax, %rdi
    negq %rbx
    movq 56(%rsp), %rsi
    addq %rbx, %rsi
    movq 32(%rsp), %rcx
    negq %rcx
    movq 40(%rsp), %rdx
    addq %rcx, %rdx
    addq %rdx, %rsi
    movq 8(%rsp), %rdx
    negq %rdx
    movq %rbp, %rcx
    addq %rdx, %rcx
    movq %rsi, %rdx
    addq %rcx, %rdx
    movq 16(%rsp), %rsi
    negq %rsi
    movq 0(%rsp), %rcx
    addq %rsi, %rcx
    movq %rdx, %rsi
    addq %rcx, %rsi
    movq %r14, %rdx
    negq %rdx
    movq 48(%rsp), %rcx
    addq %rdx, %rcx
    addq %rcx, %rsi
    movq 24(%rsp), %rdx
    negq %rdx
    movq %r13, %rcx
    addq %rdx, %rcx
    addq %rcx, %rsi
    negq %r12
    movq 64(%rsp), %rcx
    addq %r12, %rcx
    addq %rcx, %rsi
    movq %rdi, %rcx
    negq %rcx
    movq 72(%rsp), %rdx
    addq %rcx, %rdx
    movq %rsi, %rcx
    addq %rdx, %rcx
    addq $42, %rcx
    movq %rcx, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $96, %rsp
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    popq %r14
    retq 


