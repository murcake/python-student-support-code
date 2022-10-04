	.globl main
	.align 16
main:
    pushq %rbx
    subq $16, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, %rcx
    cmpq $1, %rbx
    jl block_4
    cmpq $2, %rbx
    je block_1
    jmp block_2

	.align 16
block_1:
    addq $2, %rcx
    jmp block_3

	.align 16
block_2:
    addq $10, %rcx
    jmp block_3

	.align 16
block_3:
    movq %rcx, %rdi
    callq print_int
    movq $0, %rax

	.align 16
conclusion:
    addq $16, %rsp
    popq %rbx
    retq 

	.align 16
block_4:
    cmpq $0, %rbx
    je block_1
    jmp block_2

	.align 16
block_5:
    cmpq $2, %rbx
    je block_1
    jmp block_2

	.align 16
block_6:


