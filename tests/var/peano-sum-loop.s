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
    jmp block_38

	.align 16
block_38:
    movq $0, %rax
    cmpq %rax, 0(%rsp)
    jne block_39
    jmp block_41

	.align 16
block_39:
    movq $1, %rax
    addq %rax, 8(%rsp)
    movq $1, %rax
    subq %rax, 0(%rsp)
    jmp block_38

	.align 16
block_41:
    movq 8(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $24, %rsp
    retq 


