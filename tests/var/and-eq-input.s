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
    movq $0, %rax
    cmpq %rax, 0(%rsp)
    je block_57
    jmp block_55

	.align 16
block_54:
    movq $0, %rax
    movq %rax, 8(%rsp)
    jmp block_56

	.align 16
block_55:
    movq $42, %rax
    movq %rax, 8(%rsp)
    jmp block_56

	.align 16
block_56:
    movq 8(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $24, %rsp
    retq 

	.align 16
block_57:
    callq read_int
    movq %rax, 0(%rsp)
    movq $1, %rax
    cmpq %rax, 0(%rsp)
    je block_54
    jmp block_55

	.align 16
block_59:


