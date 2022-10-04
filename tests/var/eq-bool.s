	.globl main
	.align 16
main:
    subq $8, %rsp
    jmp start

	.align 16
start:
    movq $0, %rax
    cmpq $0, %rax
    je block_1
    movq $777, %rdi
    jmp block_3

	.align 16
block_1:
    movq $42, %rdi
    jmp block_3

	.align 16
block_2:
    movq $777, %rdi
    jmp block_3

	.align 16
block_3:
    callq print_int
    movq $0, %rax

	.align 16
conclusion:
    addq $8, %rsp
    retq 


