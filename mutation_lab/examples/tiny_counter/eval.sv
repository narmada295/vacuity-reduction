module tiny_evaluation (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic active,
    input logic done
);
    EVAL_NO_EARLY_DONE: assert property (@(posedge clk) disable iff (!rst_n)
        (start && !active) |=> !done ##1 !done ##1 !done);
endmodule

bind tiny_counter tiny_evaluation evaluation_inst (.*);
