	.globl main
	.align 16
main:
    pushq %rbx
    pushq %r12
    subq $8, %rsp
    jmp start

	.align 16
start:
    movq $0, %r12
    movq $0, %rbx
    jmp block_1

	.align 16
block_1:
    cmpq $5, %r12
    jl block_2
    jmp block_4

	.align 16
block_2:
    addq %r12, %rbx
    addq $1, %r12
    jmp block_1

	.align 16
block_4:
    movq %rbx, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $8, %rsp
    popq %r12
    popq %rbx
    retq 


