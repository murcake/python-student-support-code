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
    movq $0, %rax
    movq %rax, 8(%rsp)
    jmp block_117

	.align 16
block_117:
    movq $5, %rax
    cmpq %rax, 0(%rsp)
    jl block_118
    jmp block_120

	.align 16
block_118:
    movq 0(%rsp), %rax
    addq %rax, 8(%rsp)
    movq $1, %rax
    addq %rax, 0(%rsp)
    jmp block_117

	.align 16
block_120:
    movq 8(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $24, %rsp
    retq 


