	.globl main
	.align 16
main:
    pushq %rbx
    subq $16, %rsp
    jmp start

	.align 16
start:
    movq $0, %rax
    cmpq $0, %rax
    je block_1
    movq $777, %rbx
    jmp block_3

	.align 16
block_1:
    movq $42, %rbx
    jmp block_3

	.align 16
block_2:
    movq $777, %rbx
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


