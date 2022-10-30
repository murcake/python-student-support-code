	.globl main
	.align 16
main:
    subq $24, %rsp
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
    movq %rax, 8(%rsp)
    callq read_int
    movq %rax, 8(%rsp)
    movq 0(%rsp), %rax
    addq %rax, 8(%rsp)
    movq 8(%rsp), %rax
    movq %rax, 16(%rsp)
    movq $2, %rax
    addq %rax, 16(%rsp)
    movq 16(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $24, %rsp
    retq 


