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
    movq $1, %rax
    cmpq %rax, 0(%rsp)
    jl block_45
    movq $2, %rax
    cmpq %rax, 0(%rsp)
    je block_42
    jmp block_43

	.align 16
block_42:
    movq $2, %rax
    addq %rax, 8(%rsp)
    jmp block_44

	.align 16
block_43:
    movq $10, %rax
    addq %rax, 8(%rsp)
    jmp block_44

	.align 16
block_44:
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
block_45:
    movq $0, %rax
    cmpq %rax, 0(%rsp)
    je block_42
    jmp block_43

	.align 16
block_46:
    movq $2, %rax
    cmpq %rax, 0(%rsp)
    je block_42
    jmp block_43

	.align 16
block_47:


