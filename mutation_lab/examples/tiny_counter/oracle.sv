module tiny_oracle (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic active,
    input logic done
);
    ORACLE_LATENCY: assert property (@(posedge clk) disable iff (!rst_n)
        (start && !active) |=> !done ##1 !done ##1 !done ##1 done);
    REACH_REQUEST: cover property (@(posedge clk)
        rst_n && start && !active);
endmodule

bind tiny_counter tiny_oracle oracle_inst (.*);
