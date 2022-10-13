	.globl main
	.align 16
main:
    pushq %r14
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    subq $32, %rsp
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, %r14
    callq read_int
    movq %rax, 0(%rsp)
    cmpq 0(%rsp), %r14
    jne block_4
    jmp block_2

	.align 16
block_1:
    cmpq 0(%rsp), %r14
    setl %al
    movzbq %al, %rbx
    cmpq 0(%rsp), %r14
    setl %al
    movzbq %al, %r12
    cmpq 0(%rsp), %r14
    setg %al
    movzbq %al, %r12
    cmpq 0(%rsp), %r14
    setge %al
    movzbq %al, %r12
    jmp block_3

	.align 16
block_2:
    movq $1, %rbx
    movq $1, %r12
    movq $1, %r12
    movq $1, %r12
    jmp block_3

	.align 16
block_3:
    cmpq $1, %rbx
    je block_37
    jmp block_32

	.align 16
block_4:
    cmpq 0(%rsp), %r14
    je block_1
    jmp block_1

	.align 16
block_6:

	.align 16
block_12:

	.align 16
block_23:
    cmpq 0(%rsp), %r14
    jne block_4
    jmp block_2

	.align 16
block_24:

	.align 16
block_31:
    cmpq $1, %r12
    je block_34
    movq $0, 8(%rsp)
    jmp block_36

	.align 16
block_32:
    movq $0, %rbp
    jmp block_33

	.align 16
block_33:
    cmpq $1, %rbp
    je block_47
    movq %r14, %rcx
    subq 0(%rsp), %rcx
    movq %rcx, %r13
    jmp block_48

	.align 16
block_34:
    movq %r12, 8(%rsp)
    jmp block_36

	.align 16
block_35:
    movq $0, 8(%rsp)
    jmp block_36

	.align 16
block_36:
    movq 8(%rsp), %rbp
    jmp block_33

	.align 16
block_37:
    cmpq $1, %r12
    je block_31
    jmp block_32

	.align 16
block_39:

	.align 16
block_46:
    movq %r14, %rcx
    subq 0(%rsp), %rcx
    movq %rcx, %r13
    jmp block_48

	.align 16
block_47:
    movq $0, %r13
    jmp block_48

	.align 16
block_48:
    movq %r13, %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    addq $32, %rsp
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    popq %r14
    retq 


