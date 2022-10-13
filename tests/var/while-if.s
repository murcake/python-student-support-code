	.globl main
	.align 16
main:
    pushq %rbx
    subq $16, %rsp
    jmp start

	.align 16
start:
    movq $0, %rbx
    jmp block_1

	.align 16
block_1:
    callq read_int
    movq %rax, %rcx
    cmpq $5, %rcx
    je block_2
    jmp block_4

	.align 16
block_2:
    addq $42, %rbx
    jmp block_1

	.align 16
block_4:
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
block_7:


