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
    movq $0, %rax
    movq %rax, 0(%rsp)
    jmp block_25

	.align 16
block_25:
    callq read_int
    movq %rax, 8(%rsp)
    movq $5, %rax
    cmpq %rax, 8(%rsp)
    je block_26
    jmp block_28

	.align 16
block_26:
    movq $42, %rax
    addq %rax, 0(%rsp)
    jmp block_25

	.align 16
block_28:
    movq 0(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $24, %rsp
    retq 

	.align 16
block_31:


