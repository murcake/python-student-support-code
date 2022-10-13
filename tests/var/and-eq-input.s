	.globl main
	.align 16
main:
    pushq %rbx
    subq $16, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, %rcx
    cmpq $0, %rcx
    je block_4
    jmp block_2

	.align 16
block_1:
    movq $0, %rbx
    jmp block_3

	.align 16
block_2:
    movq $42, %rbx
    jmp block_3

	.align 16
block_3:
    movq %rbx, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $16, %rsp
    popq %rbx
    retq 

	.align 16
block_4:
    callq read_int
    movq %rax, %rcx
    cmpq $1, %rcx
    je block_1
    jmp block_2

	.align 16
block_6:


