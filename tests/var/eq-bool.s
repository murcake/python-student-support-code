	.globl main
	.align 16
main:
    subq $8, %rsp
    movq $16384, %rdi
    movq $16384, %rsi
    callq initialize
    movq rootstack_begin(%rip), %r15
    addq $0, %r15
    jmp start

	.align 16
start:
    movq $0, %rax
    cmpq $0, %rax
    je block_114
    movq $777, %rax
    movq %rax, 0(%rsp)
    jmp block_116

	.align 16
block_114:
    movq $42, %rax
    movq %rax, 0(%rsp)
    jmp block_116

	.align 16
block_115:
    movq $777, %rax
    movq %rax, 0(%rsp)
    jmp block_116

	.align 16
block_116:
    movq 0(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $8, %rsp
    retq 


