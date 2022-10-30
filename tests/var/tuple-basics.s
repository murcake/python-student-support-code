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
    movq $0, 24(%r15)
    addq $4, %r15
    jmp start

	.align 16
start:
    movq $1, %rax
    movq %rax, 0(%rsp)
    movq free_ptr(%rip), %rax
    movq %rax, 8(%rsp)
    movq $16, %rax
    addq %rax, 8(%rsp)
    movq fromspace_end(%rip), %rax
    movq %rax, 16(%rsp)
    movq 16(%rsp), %rax
    cmpq %rax, 8(%rsp)
    jge block_1
    jmp block_3

	.align 16
block_1:
    movq %r15, %rdi
    movq $16, %rsi
    callq collect
    jmp block_3

	.align 16
block_3:
    movq free_ptr(%rip), %r11
    movq $16, %rax
    addq %rax, free_ptr(%rip)
    movq $2, %rax
    movq %rax, 0(%r11)
    movq %r11, -8(%r15)
    movq -8(%r15), %r11
    movq 0(%rsp), %rax
    movq %rax, 8(%r11)
    movq -8(%r15), %r11
    movq 8(%r11), %rax
    movq %rax, 24(%rsp)
    movq 24(%rsp), %rdi
    callq print_int
    movq $1, %rax
    movq %rax, 32(%rsp)
    movq $1, %rax
    movq %rax, 0(%rsp)
    movq $0, %rax
    movq %rax, 40(%rsp)
    movq free_ptr(%rip), %rax
    movq %rax, 8(%rsp)
    movq $32, %rax
    addq %rax, 8(%rsp)
    movq fromspace_end(%rip), %rax
    movq %rax, 16(%rsp)
    movq 16(%rsp), %rax
    cmpq %rax, 8(%rsp)
    jge block_4
    jmp block_6

	.align 16
block_4:
    movq %r15, %rdi
    movq $32, %rsi
    callq collect
    jmp block_6

	.align 16
block_6:
    movq free_ptr(%rip), %r11
    movq $32, %rax
    addq %rax, free_ptr(%rip)
    movq $6, %rax
    movq %rax, 0(%r11)
    movq %r11, -16(%r15)
    movq -16(%r15), %r11
    movq 32(%rsp), %rax
    movq %rax, 8(%r11)
    movq -16(%r15), %r11
    movq 0(%rsp), %rax
    movq %rax, 16(%r11)
    movq -16(%r15), %r11
    movq 40(%rsp), %rax
    movq %rax, 24(%r11)
    movq -16(%r15), %rax
    movq %rax, -24(%r15)
    movq -24(%r15), %r11
    movq 24(%r11), %rax
    movq %rax, 16(%rsp)
    movq $1, %rax
    cmpq %rax, 16(%rsp)
    je block_7
    movq $0, %rax
    movq %rax, 24(%rsp)
    jmp block_9

	.align 16
block_7:
    movq $1, %rax
    movq %rax, 24(%rsp)
    jmp block_9

	.align 16
block_8:
    movq $0, %rax
    movq %rax, 24(%rsp)
    jmp block_9

	.align 16
block_9:
    movq 24(%rsp), %rdi
    callq print_int
    movq $1, %rax
    movq %rax, 32(%rsp)
    movq -24(%r15), %rax
    movq %rax, -32(%r15)
    movq -8(%r15), %rax
    movq %rax, -24(%r15)
    movq free_ptr(%rip), %rax
    movq %rax, 16(%rsp)
    movq $32, %rax
    addq %rax, 16(%rsp)
    movq fromspace_end(%rip), %rax
    movq %rax, 8(%rsp)
    movq 8(%rsp), %rax
    cmpq %rax, 16(%rsp)
    jge block_10
    jmp block_12

	.align 16
block_10:
    movq %r15, %rdi
    movq $32, %rsi
    callq collect
    jmp block_12

	.align 16
block_12:
    movq free_ptr(%rip), %r11
    movq $32, %rax
    addq %rax, free_ptr(%rip)
    movq $774, %rax
    movq %rax, 0(%r11)
    movq %r11, -16(%r15)
    movq -16(%r15), %r11
    movq 32(%rsp), %rax
    movq %rax, 8(%r11)
    movq -16(%r15), %r11
    movq -32(%r15), %rax
    movq %rax, 16(%r11)
    movq -16(%r15), %r11
    movq -24(%r15), %rax
    movq %rax, 24(%r11)
    movq -16(%r15), %r11
    movq 16(%r11), %rax
    movq %rax, -16(%r15)
    movq -16(%r15), %r11
    movq 16(%r11), %rax
    movq %rax, 24(%rsp)
    movq 24(%rsp), %rdi
    callq print_int
    movq $0, %rax
    jmp conclusion

	.align 16
conclusion:
    subq $4, %r15
    addq $48, %rsp
    popq %r15
    retq 


