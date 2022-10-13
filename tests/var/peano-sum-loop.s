	.globl main
	.align 16
main:
    pushq %rbx
    pushq %r12
    subq $8, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, %rbx
    callq read_int
    movq %rax, %r12
    jmp block_1

	.align 16
block_1:
    cmpq $0, %rbx
    jne block_2
    jmp block_4

	.align 16
block_2:
    addq $1, %r12
    subq $1, %rbx
    jmp block_1

	.align 16
block_4:
    movq %r12, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $8, %rsp
    popq %r12
    popq %rbx
    retq 


