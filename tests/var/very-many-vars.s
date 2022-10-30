	.globl main
	.align 16
main:
    subq $168, %rsp
    movq $16384, %rdi
    movq $16384, %rsi
    callq initialize
    movq rootstack_begin(%rip), %r15
    addq $0, %r15
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, 0(%rsp)
    callq read_int
    movq %rax, 8(%rsp)
    callq read_int
    movq %rax, 16(%rsp)
    callq read_int
    movq %rax, 24(%rsp)
    callq read_int
    movq %rax, 32(%rsp)
    callq read_int
    movq %rax, 40(%rsp)
    callq read_int
    movq %rax, 48(%rsp)
    callq read_int
    movq %rax, 56(%rsp)
    callq read_int
    movq %rax, 64(%rsp)
    callq read_int
    movq %rax, 72(%rsp)
    callq read_int
    movq %rax, 80(%rsp)
    callq read_int
    movq %rax, 88(%rsp)
    callq read_int
    movq %rax, 96(%rsp)
    callq read_int
    movq %rax, 104(%rsp)
    callq read_int
    movq %rax, 112(%rsp)
    callq read_int
    movq %rax, 120(%rsp)
    negq 8(%rsp)
    movq 8(%rsp), %rax
    addq %rax, 0(%rsp)
    movq 24(%rsp), %rax
    movq %rax, 128(%rsp)
    negq 128(%rsp)
    movq 16(%rsp), %rax
    movq %rax, 136(%rsp)
    movq 128(%rsp), %rax
    addq %rax, 136(%rsp)
    movq 136(%rsp), %rax
    addq %rax, 0(%rsp)
    movq 40(%rsp), %rax
    movq %rax, 128(%rsp)
    negq 128(%rsp)
    movq 32(%rsp), %rax
    movq %rax, 136(%rsp)
    movq 128(%rsp), %rax
    addq %rax, 136(%rsp)
    movq 136(%rsp), %rax
    addq %rax, 0(%rsp)
    movq 56(%rsp), %rax
    movq %rax, 136(%rsp)
    negq 136(%rsp)
    movq 48(%rsp), %rax
    movq %rax, 128(%rsp)
    movq 136(%rsp), %rax
    addq %rax, 128(%rsp)
    movq 0(%rsp), %rax
    movq %rax, 136(%rsp)
    movq 128(%rsp), %rax
    addq %rax, 136(%rsp)
    movq 72(%rsp), %rax
    movq %rax, 144(%rsp)
    negq 144(%rsp)
    movq 64(%rsp), %rax
    movq %rax, 128(%rsp)
    movq 144(%rsp), %rax
    addq %rax, 128(%rsp)
    movq 128(%rsp), %rax
    addq %rax, 136(%rsp)
    movq 88(%rsp), %rax
    movq %rax, 128(%rsp)
    negq 128(%rsp)
    movq 128(%rsp), %rax
    addq %rax, 80(%rsp)
    movq 136(%rsp), %rax
    movq %rax, 128(%rsp)
    movq 80(%rsp), %rax
    addq %rax, 128(%rsp)
    movq 104(%rsp), %rax
    movq %rax, 136(%rsp)
    negq 136(%rsp)
    movq 96(%rsp), %rax
    movq %rax, 152(%rsp)
    movq 136(%rsp), %rax
    addq %rax, 152(%rsp)
    movq 128(%rsp), %rax
    movq %rax, 144(%rsp)
    movq 152(%rsp), %rax
    addq %rax, 144(%rsp)
    movq 120(%rsp), %rax
    movq %rax, 128(%rsp)
    negq 128(%rsp)
    movq 112(%rsp), %rax
    movq %rax, 136(%rsp)
    movq 128(%rsp), %rax
    addq %rax, 136(%rsp)
    movq 136(%rsp), %rax
    addq %rax, 144(%rsp)
    movq $42, %rax
    addq %rax, 144(%rsp)
    movq 144(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $168, %rsp
    retq 


