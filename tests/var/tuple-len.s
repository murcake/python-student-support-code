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
    addq $2, %r15
    jmp start

	.align 16
start:
    movq $38, %rax
    movq %rax, 0(%rsp)
    movq $2, %rax
    movq %rax, 8(%rsp)
    movq free_ptr(%rip), %rax
    movq %rax, 16(%rsp)
    movq $24, %rax
    addq %rax, 16(%rsp)
    movq fromspace_end(%rip), %rax
    movq %rax, 24(%rsp)
    movq 24(%rsp), %rax
    cmpq %rax, 16(%rsp)
    jge block_13
    jmp block_15

	.align 16
block_13:
    movq %r15, %rdi
    movq $24, %rsi
    callq collect
    jmp block_15

	.align 16
block_15:
    movq free_ptr(%rip), %r11
    movq $24, %rax
    addq %rax, free_ptr(%rip)
    movq $4, %rax
    movq %rax, 0(%r11)
    movq %r11, -8(%r15)
    movq -8(%r15), %r11
    movq 0(%rsp), %rax
    movq %rax, 8(%r11)
    movq -8(%r15), %r11
    movq 8(%rsp), %rax
    movq %rax, 16(%r11)
    movq -8(%r15), %rax
    movq %rax, -16(%r15)
    movq -16(%r15), %r11
    movq 8(%r11), %rax
    movq %rax, 24(%rsp)
    movq -16(%r15), %r11
    movq 16(%r11), %rax
    movq %rax, 16(%rsp)
    movq 16(%rsp), %rax
    addq %rax, 24(%rsp)
    movq -16(%r15), %r11
    movq 0(%r11), %r11
    sarq $1, %r11
    andq $63, %r11
    movq %r11, 16(%rsp)
    movq 24(%rsp), %rax
    movq %rax, 32(%rsp)
    movq 16(%rsp), %rax
    addq %rax, 32(%rsp)
    movq 32(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $2, %r15
    addq $48, %rsp
    popq %r15
    retq 


