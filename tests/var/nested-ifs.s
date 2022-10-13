	.globl main
	.align 16
main:
    pushq %rbx
    pushq %r12
    pushq %r13
    subq $16, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, %r13
    callq read_int
    movq %rax, %rbx
    cmpq $1, %r13
    jl block_4
    cmpq $2, %r13
    je block_1
    jmp block_2

	.align 16
block_1:
    movq %rbx, %rcx
    addq $2, %rcx
    movq %rcx, %r12
    jmp block_3

	.align 16
block_2:
    movq %rbx, %rcx
    addq $10, %rcx
    movq %rcx, %r12
    jmp block_3

	.align 16
block_3:
    movq %r12, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $16, %rsp
    popq %r13
    popq %r12
    popq %rbx
    retq 

	.align 16
block_4:
    cmpq $0, %r13
    je block_1
    jmp block_2

	.align 16
block_5:
    cmpq $2, %r13
    je block_1
    jmp block_2

	.align 16
block_6:


