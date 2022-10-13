	.globl main
	.align 16
main:
    subq $8, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $8, %rsp
    retq 


