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
    callq read_int
    movq %rax, %rcx
    callq read_int
    movq %rax, %rcx
    addq %rbx, %rcx
    movq %rcx, %rdi
    addq $2, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $16, %rsp
    popq %rbx
    retq 


