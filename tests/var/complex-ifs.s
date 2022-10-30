	.globl main
	.align 16
main:
    subq $56, %rsp
    movq $16384, %rdi
    movq $16384, %rsi
    callq initialize
    movq rootstack_begin(%rip), %r15
    addq $0, %r15
    jmp start

	.align 16
start:
    callq read_int
    movq %rax, 0(%rsp)
    callq read_int
    movq %rax, 8(%rsp)
    movq 8(%rsp), %rax
    cmpq %rax, 0(%rsp)
    jne block_69
    jmp block_67

	.align 16
block_66:
    movq 8(%rsp), %rax
    cmpq %rax, 0(%rsp)
    setl %al
    movzbq %al, %rax
    movq %rax, 16(%rsp)
    movq 8(%rsp), %rax
    cmpq %rax, 0(%rsp)
    setl %al
    movzbq %al, %rax
    movq %rax, 24(%rsp)
    movq 8(%rsp), %rax
    cmpq %rax, 0(%rsp)
    setg %al
    movzbq %al, %rax
    movq %rax, 32(%rsp)
    movq 8(%rsp), %rax
    cmpq %rax, 0(%rsp)
    setge %al
    movzbq %al, %rax
    movq %rax, 40(%rsp)
    jmp block_68

	.align 16
block_67:
    movq $1, %rax
    movq %rax, 16(%rsp)
    movq $1, %rax
    movq %rax, 24(%rsp)
    movq $1, %rax
    movq %rax, 32(%rsp)
    movq $1, %rax
    movq %rax, 40(%rsp)
    jmp block_68

	.align 16
block_68:
    movq $1, %rax
    cmpq %rax, 16(%rsp)
    je block_102
    jmp block_97

	.align 16
block_69:
    movq 8(%rsp), %rax
    cmpq %rax, 0(%rsp)
    je block_66
    jmp block_66

	.align 16
block_71:

	.align 16
block_77:

	.align 16
block_88:
    movq 8(%rsp), %rax
    cmpq %rax, 0(%rsp)
    jne block_69
    jmp block_67

	.align 16
block_89:

	.align 16
block_96:
    movq $1, %rax
    cmpq %rax, 32(%rsp)
    je block_99
    movq $0, %rax
    movq %rax, 32(%rsp)
    jmp block_101

	.align 16
block_97:
    movq $0, %rax
    movq %rax, 32(%rsp)
    jmp block_98

	.align 16
block_98:
    movq $1, %rax
    cmpq %rax, 32(%rsp)
    je block_112
    movq 0(%rsp), %rax
    movq %rax, 40(%rsp)
    movq 8(%rsp), %rax
    subq %rax, 40(%rsp)
    jmp block_113

	.align 16
block_99:
    movq 40(%rsp), %rax
    movq %rax, 32(%rsp)
    jmp block_101

	.align 16
block_100:
    movq $0, %rax
    movq %rax, 32(%rsp)
    jmp block_101

	.align 16
block_101:
    jmp block_98

	.align 16
block_102:
    movq $1, %rax
    cmpq %rax, 24(%rsp)
    je block_96
    jmp block_97

	.align 16
block_104:

	.align 16
block_111:
    movq 0(%rsp), %rax
    movq %rax, 40(%rsp)
    movq 8(%rsp), %rax
    subq %rax, 40(%rsp)
    jmp block_113

	.align 16
block_112:
    movq $0, %rax
    movq %rax, 40(%rsp)
    jmp block_113

	.align 16
block_113:
    movq 40(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $0, %r15
    addq $56, %rsp
    retq 


