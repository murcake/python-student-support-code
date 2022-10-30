	.globl main
	.align 16
main:
    pushq %r15
    subq $48, %rsp
    movq $16384, %rdi
    movq $16384, %rsi
    callq initialize
    movq rootstack_begin(%rip), %r15
    movq $0, 0(%r15)
    movq $0, 8(%r15)
    movq $0, 16(%r15)
    addq $3, %r15
    jmp start

	.align 16
start:
    movq $40, %rax
    movq %rax, 0(%rsp)
    movq $1, %rax
    movq %rax, 8(%rsp)
    movq $2, %rax
    movq %rax, 16(%rsp)
    movq free_ptr(%rip), %rax
    movq %rax, 24(%rsp)
    movq $16, %rax
    addq %rax, 24(%rsp)
    movq fromspace_end(%rip), %rax
    movq %rax, 32(%rsp)
    movq 32(%rsp), %rax
    cmpq %rax, 24(%rsp)
    jge block_16
    jmp block_18

	.align 16
block_16:
    movq %r15, %rdi
    movq $16, %rsi
    callq collect
    jmp block_18

	.align 16
block_18:
    movq free_ptr(%rip), %r11
    movq $16, %rax
    addq %rax, free_ptr(%rip)
    movq $2, %rax
    movq %rax, 0(%r11)
    movq %r11, -8(%r15)
    movq -8(%r15), %r11
    movq 16(%rsp), %rax
    movq %rax, 8(%r11)
    movq -8(%r15), %rax
    movq %rax, -16(%r15)
    movq free_ptr(%rip), %rax
    movq %rax, 24(%rsp)
    movq $32, %rax
    addq %rax, 24(%rsp)
    movq fromspace_end(%rip), %rax
    movq %rax, 32(%rsp)
    movq 32(%rsp), %rax
    cmpq %rax, 24(%rsp)
    jge block_19
    jmp block_21

	.align 16
block_19:
    movq %r15, %rdi
    movq $32, %rsi
    callq collect
    jmp block_21

	.align 16
block_21:
    movq free_ptr(%rip), %r11
    movq $32, %rax
    addq %rax, free_ptr(%rip)
    movq $518, %rax
    movq %rax, 0(%r11)
    movq %r11, -8(%r15)
    movq -8(%r15), %r11
    movq 0(%rsp), %rax
    movq %rax, 8(%r11)
    movq -8(%r15), %r11
    movq 8(%rsp), %rax
    movq %rax, 16(%r11)
    movq -8(%r15), %r11
    movq -16(%r15), %rax
    movq %rax, 24(%r11)
    movq -8(%r15), %rax
    movq %rax, -24(%r15)
    movq -24(%r15), %r11
    movq 16(%r11), %rax
    movq %rax, 24(%rsp)
    movq $1, %rax
    cmpq %rax, 24(%rsp)
    je block_22
    movq $44, %rax
    movq %rax, 32(%rsp)
    jmp block_24

	.align 16
block_22:
    movq -24(%r15), %r11
    movq 8(%r11), %rax
    movq %rax, 32(%rsp)
    movq -24(%r15), %r11
    movq 24(%r11), %rax
    movq %rax, -8(%r15)
    movq -8(%r15), %r11
    movq 8(%r11), %rax
    movq %rax, 24(%rsp)
    movq 24(%rsp), %rax
    addq %rax, 32(%rsp)
    jmp block_24

	.align 16
block_23:
    movq $44, %rax
    movq %rax, 32(%rsp)
    jmp block_24

	.align 16
block_24:
    movq 32(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $3, %r15
    addq $48, %rsp
    popq %r15
    retq 


