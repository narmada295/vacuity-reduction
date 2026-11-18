module tiny_counter (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done
);
    logic active;
    logic [1:0] count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            count  <= 2'd0;
            done   <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && !active) begin
                active <= 1'b1;
                count <= 2'd2;
            end else if (active) begin
                if (count == 2'd0) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end else begin
                    count <= count - 2'd1;
                end
            end
        end
    end
endmodule
