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
    cmpq %rcx, %rbx
    jne block_4
    jmp block_2

	.align 16
block_1:
    cmpq %rcx, %rbx
    setl %al
    movzbq %al, %rdi
    cmpq %rcx, %rbx
    setl %al
    movzbq %al, %r8
    cmpq %rcx, %rbx
    setg %al
    movzbq %al, %rsi
    cmpq %rcx, %rbx
    setge %al
    movzbq %al, %rdx
    jmp block_3

	.align 16
block_2:
    movq $1, %rdi
    movq $1, %r8
    movq $1, %rsi
    movq $1, %rdx
    jmp block_3

	.align 16
block_3:
    cmpq $1, %rdi
    je block_37
    jmp block_32

	.align 16
block_4:
    cmpq %rcx, %rbx
    je block_1
    jmp block_1

	.align 16
block_6:

	.align 16
block_12:

	.align 16
block_23:
    cmpq %rcx, %rbx
    jne block_4
    jmp block_2

	.align 16
block_24:

	.align 16
block_31:
    cmpq $1, %rsi
    je block_34
    movq $0, %rdx
    jmp block_36

	.align 16
block_32:
    movq $0, %rdx
    jmp block_33

	.align 16
block_33:
    cmpq $1, %rdx
    je block_47
    subq %rcx, %rbx
    movq %rbx, %rdi
    jmp block_48

	.align 16
block_34:
    jmp block_36

	.align 16
block_35:
    movq $0, %rdx
    jmp block_36

	.align 16
block_36:
    jmp block_33

	.align 16
block_37:
    cmpq $1, %r8
    je block_31
    jmp block_32

	.align 16
block_39:

	.align 16
block_46:
    subq %rcx, %rbx
    movq %rbx, %rdi
    jmp block_48

	.align 16
block_47:
    movq $0, %rdi
    jmp block_48

	.align 16
block_48:
    callq print_int
    movq $0, %rax

	.align 16
conclusion:
    addq $16, %rsp
    popq %rbx
    retq 


