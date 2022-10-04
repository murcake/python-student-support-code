	.globl main
	.align 16
main:
    subq $8, %rsp
    jmp start

	.align 16
start:
    movq $0, %rdi
    callq print_int
    movq $0, %rax

	.align 16
conclusion:
    addq $8, %rsp
    retq 


