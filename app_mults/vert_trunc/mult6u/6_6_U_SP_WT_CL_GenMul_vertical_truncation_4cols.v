//Compilation time: 2024-06-03 22:50:07
//Compilation SHA256 message digest: df284b25599f92449be9e44000d51942df9212fd959aee6c8c35c8a3184951b7
/*----------------------------------------------------------------------------
Copyright (c) 2019-2020 University of Bremen, Germany.
Copyright (c) 2020 Johannes Kepler University Linz, Austria.
This file has been generated with GenMul.
You can find GenMul at: http://www.sca-verification.org/genmul
Contact us at genmul@sca-verification.org

  First input length: 6
  second input length: 6
  Partial product generator: Unsigned simple partial product generator [U_SP]
  Partial product accumulator: Wallace tree [WT]
  Final stage adder: Carry look-ahead adder [CL]
----------------------------------------------------------------------------*/
module FullAdder(X, Y, Z, S, C);
  output C;
  output S;
  input X;
  input Y;
  input Z;
  assign C = ( X & Y ) | ( Y & Z ) | ( Z & X );
  assign S = X ^ Y ^ Z;
endmodule
module FullAdderProp(X, Y, Z, S, C, P);
  output C;
  output S;
  output P;
  input X;
  input Y;
  input Z;
  assign C = ( X & Y ) | ( Y & Z ) | ( Z & X );
  assign S = X ^ Y ^ Z;
  assign P = X ^ Y;
endmodule
module HalfAdder(X, Y, S, C);
  output C;
  output S;
  input X;
  input Y;
  assign C = X & Y;
  assign S = X ^ Y;
endmodule
module ConstatntOne(O);
  output O;
  assign O = 1;
endmodule
module Counter(X1, X2, X3, X4, X5, X6, X7, S3, S2, S1);
output S1;
output S2;
output S3;
input X1;
input X2;
input X3;
input X4;
input X5;
input X6;
input X7;
wire W1;
wire W2;
wire W3;
wire W4;
wire W5;
wire W6;
assign W1 = X1 ^ X2 ^ X3;
assign W2 = X4 ^ X5 ^ ( X6 ^ X7 );
assign W3 = ~ ( ( ~ ( X1 & X2 ) ) & ( ~ ( X1 & X3 ) ) & ( ~ ( X2 & X3 ) ) );
assign W4 = ~ ( ( ~ ( ( X4 | X5 ) & ( X6 | X7 ) ) ) & ( ~ ( ( X4 & X5 ) | ( X6 & X7 ) ) ) );
assign W5 = ~ ( X4 & X5 & X6 & X7 );
assign W6 = ~ ( ( ~ ( W4 & W5 ) ) ^ W3 );
assign S3 = W1 ^ W2;
assign S2 = ~ ( W6 ^ ( ~ ( W1 & W2 ) ) );
assign S1 = ~ ( W5 & ( ~ ( W3 & W4 ) ) & ( ~ ( W1 & W2 & W6 ) ) );
endmodule
module U_SP_6_6(IN1, IN2 , P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
  input [5:0] IN1;
  input [5:0] IN2;
  output [0:0] P0;
  output [1:0] P1;
  output [2:0] P2;
  output [3:0] P3;
  output [4:0] P4;
  output [5:0] P5;
  output [4:0] P6;
  output [3:0] P7;
  output [2:0] P8;
  output [1:0] P9;
  output [0:0] P10;
  // assign P0[0] = IN1[0]&IN2[0];
  // assign P1[0] = IN1[0]&IN2[1];
  // assign P2[0] = IN1[0]&IN2[2];
  // assign P3[0] = IN1[0]&IN2[3];
  assign P0[0] = 1'b0;
  assign P1[0] = 1'b0;
  assign P2[0] = 1'b0;
  assign P3[0] = 1'b0;
  assign P4[0] = IN1[0]&IN2[4];
  assign P5[0] = IN1[0]&IN2[5];
  // assign P1[1] = IN1[1]&IN2[0];
  // assign P2[1] = IN1[1]&IN2[1];
  // assign P3[1] = IN1[1]&IN2[2];
  assign P1[1] = 1'b0; 
  assign P2[1] = 1'b0; 
  assign P3[1] = 1'b0; 
  assign P4[1] = IN1[1]&IN2[3];
  assign P5[1] = IN1[1]&IN2[4];
  assign P6[0] = IN1[1]&IN2[5];
  // assign P2[2] = IN1[2]&IN2[0];
  // assign P3[2] = IN1[2]&IN2[1];
  assign P2[2] = 1'b0;
  assign P3[2] = 1'b0;
  assign P4[2] = IN1[2]&IN2[2];
  assign P5[2] = IN1[2]&IN2[3];
  assign P6[1] = IN1[2]&IN2[4];
  assign P7[0] = IN1[2]&IN2[5];
  // assign P3[3] = IN1[3]&IN2[0];
  assign P3[3] = 1'b0;
  assign P4[3] = IN1[3]&IN2[1];
  assign P5[3] = IN1[3]&IN2[2];
  assign P6[2] = IN1[3]&IN2[3];
  assign P7[1] = IN1[3]&IN2[4];
  assign P8[0] = IN1[3]&IN2[5];
  assign P4[4] = IN1[4]&IN2[0];
  assign P5[4] = IN1[4]&IN2[1];
  assign P6[3] = IN1[4]&IN2[2];
  assign P7[2] = IN1[4]&IN2[3];
  assign P8[1] = IN1[4]&IN2[4];
  assign P9[0] = IN1[4]&IN2[5];
  assign P5[5] = IN1[5]&IN2[0];
  assign P6[4] = IN1[5]&IN2[1];
  assign P7[3] = IN1[5]&IN2[2];
  assign P8[2] = IN1[5]&IN2[3];
  assign P9[1] = IN1[5]&IN2[4];
  assign P10[0] = IN1[5]&IN2[5];

endmodule
module WT(IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, IN10, Out1, Out2);
  input [0:0] IN0;
  input [1:0] IN1;
  input [2:0] IN2;
  input [3:0] IN3;
  input [4:0] IN4;
  input [5:0] IN5;
  input [4:0] IN6;
  input [3:0] IN7;
  input [2:0] IN8;
  input [1:0] IN9;
  input [0:0] IN10;
  output [11:0] Out1;
  output [7:0] Out2;
  wire w37;
  wire w38;
  wire w39;
  wire w40;
  wire w41;
  wire w42;
  wire w43;
  wire w44;
  wire w45;
  wire w46;
  wire w47;
  wire w48;
  wire w49;
  wire w50;
  wire w51;
  wire w52;
  wire w53;
  wire w54;
  wire w55;
  wire w56;
  wire w57;
  wire w58;
  wire w59;
  wire w61;
  wire w62;
  wire w63;
  wire w64;
  wire w65;
  wire w66;
  wire w67;
  wire w68;
  wire w69;
  wire w70;
  wire w71;
  wire w72;
  wire w73;
  wire w74;
  wire w75;
  wire w76;
  HalfAdder U0 (IN1[0], IN1[1], Out1[1], w37);
  FullAdder U1 (IN2[0], IN2[1], IN2[2], w38, w39);
  FullAdder U2 (IN3[0], IN3[1], IN3[2], w40, w41);
  FullAdder U3 (IN4[0], IN4[1], IN4[2], w42, w43);
  HalfAdder U4 (IN4[3], IN4[4], w44, w45);
  FullAdder U5 (IN5[0], IN5[1], IN5[2], w46, w47);
  FullAdder U6 (IN5[3], IN5[4], IN5[5], w48, w49);
  FullAdder U7 (IN6[0], IN6[1], IN6[2], w50, w51);
  HalfAdder U8 (IN6[3], IN6[4], w52, w53);
  FullAdder U9 (IN7[0], IN7[1], IN7[2], w54, w55);
  FullAdder U10 (IN8[0], IN8[1], IN8[2], w56, w57);
  HalfAdder U11 (IN9[0], IN9[1], w58, w59);
  HalfAdder U12 (w37, w38, Out1[2], w61);
  FullAdder U13 (IN3[3], w39, w40, w62, w63);
  FullAdder U14 (w41, w42, w44, w64, w65);
  FullAdder U15 (w43, w45, w46, w66, w67);
  FullAdder U16 (w47, w49, w50, w68, w69);
  FullAdder U17 (IN7[3], w51, w53, w70, w71);
  HalfAdder U18 (w55, w56, w72, w73);
  HalfAdder U19 (w57, w58, w74, w75);
  HalfAdder U20 (IN10[0], w59, w76, Out1[11]);
  HalfAdder U21 (w61, w62, Out1[3], Out1[4]);
  HalfAdder U22 (w63, w64, Out2[0], Out1[5]);
  FullAdder U23 (w48, w65, w66, Out2[1], Out1[6]);
  FullAdder U24 (w52, w67, w68, Out2[2], Out1[7]);
  FullAdder U25 (w54, w69, w70, Out2[3], Out1[8]);
  HalfAdder U26 (w71, w72, Out2[4], Out1[9]);
  HalfAdder U27 (w73, w74, Out2[5], Out1[10]);
  HalfAdder U28 (w75, w76, Out2[6], Out2[7]);
  assign Out1[0] = IN0[0];

endmodule
module CL_8_8(IN1, IN2, Out);
  input [7:0] IN1;
  input [7:0] IN2;
  output [8:0] Out;
  wire w17;
  wire w18;
  wire w19;
  wire w20;
  wire w21;
  wire w22;
  wire w23;
  wire w24;
  wire w25;
  wire w26;
  wire w27;
  wire w28;
  wire w29;
  wire w30;
  wire w31;
  wire w32;
  wire w33;
  wire w34;
  wire w35;
  wire w36;
  wire w37;
  wire w38;
  HalfAdder U0 (IN1[0], IN2[0], Out[0], w17);
  HalfAdder U1 (IN1[1], IN2[1], w18, w19);
  HalfAdder U2 (IN1[2], IN2[2], w20, w21);
  HalfAdder U3 (IN1[3], IN2[3], w22, w23);
  HalfAdder U4 (IN1[4], IN2[4], w24, w25);
  HalfAdder U5 (IN1[5], IN2[5], w26, w27);
  HalfAdder U6 (IN1[6], IN2[6], w28, w29);
  HalfAdder U7 (IN1[7], IN2[7], w30, w31);
  assign Out[1] = w18^w32;
  assign Out[2] = w20^w33;
  assign Out[3] = w22^w34;
  assign Out[4] = w24^w35;
  assign Out[5] = w26^w36;
  assign Out[6] = w28^w37;
  assign Out[7] = w30^w38;

  assign w32 = (w17);
  assign w33 = (w18 & w17) | (w19);
  assign w34 = (w20 & w18 & w17) | (w20 & w19) | (w21);
  assign w35 = (w22 & w20 & w18 & w17) | (w22 & w20 & w19) | (w22 & w21) | (w23);
  assign w36 = (w24 & w22 & w20 & w18 & w17) | (w24 & w22 & w20 & w19) | (w24 & w22 & w21) | (w24 & w23) | (w25);
  assign w37 = (w26 & w24 & w22 & w20 & w18 & w17) | (w26 & w24 & w22 & w20 & w19) | (w26 & w24 & w22 & w21) | (w26 & w24 & w23) | (w26 & w25) | (w27);
  assign w38 = (w28 & w26 & w24 & w22 & w20 & w18 & w17) | (w28 & w26 & w24 & w22 & w20 & w19) | (w28 & w26 & w24 & w22 & w21) | (w28 & w26 & w24 & w23) | (w28 & w26 & w25) | (w28 & w27) | (w29);
  assign Out[8] = (w30 & w28 & w26 & w24 & w22 & w20 & w18 & w17) | (w30 & w28 & w26 & w24 & w22 & w20 & w19) | (w30 & w28 & w26 & w24 & w22 & w21) | (w30 & w28 & w26 & w24 & w23) | (w30 & w28 & w26 & w25) | (w30 & w28 & w27) | (w30 & w29) | (w31);
endmodule
module Mult_6_6(IN1, IN2, Out);
  input [5:0] IN1;
  input [5:0] IN2;
  output [11:0] Out;
  wire [0:0] P0;
  wire [1:0] P1;
  wire [2:0] P2;
  wire [3:0] P3;
  wire [4:0] P4;
  wire [5:0] P5;
  wire [4:0] P6;
  wire [3:0] P7;
  wire [2:0] P8;
  wire [1:0] P9;
  wire [0:0] P10;
  wire [11:0] R1;
  wire [7:0] R2;
  wire [12:0] aOut;
  U_SP_6_6 S0 (IN1, IN2 , P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
  WT S1 (P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, R1, R2);
  CL_8_8 S2 (R1[11:4], R2, aOut[12:4]);
  assign aOut[0] = R1[0];
  assign aOut[1] = R1[1];
  assign aOut[2] = R1[2];
  assign aOut[3] = R1[3];
  assign Out = aOut[11:0];
endmodule

/*---------------------------------------------------------------------------------------------------
This are SHA256 message digests computed for all source files to see the version of a file in genmul.
4164f841dcd2afb1341b584b40f40d82ef4c3d0830615503b45c13f9ff1f0b99 Array.cpp\nb66c13355402b785b10523902060ea2b13495e0139581e3e1f9eb511b63091d0 Array.hpp\n049f660c454752c510b8d2d7d5474b271804a3f2ef3bbd08b201b27cf5aa953c BrentKungAdder.cpp\n542006e2bd9fe43a38fbd454d79c40216e2222968ee81d61ccb8b9f5f7c2cfc0 BrentKungAdder.hpp\ndd0e0628729b9c228e1e3e2d8c759e3c5d548b2a7fc7a0ecf965f77c057f8982 CarryLookAhead.cpp\nacab3bf6b24f0596013dfce2d21a56d16bc4565d810d7521bef6b645b6192ddf CarryLookAhead.hpp\n929ba8b159d43cd1483702c39adf3f3c01a0e029393157d2dff1e576c3784ba2 CarryPredictor.cpp\nc674a6131475e399d67794664958ce3992e34ad5a8a276b258f3b65179dbb0d2 CarryPredictor.hpp\n7ad1df61ce6dec8970320cdbb0cd1b9d9e1b1aaed8337fad2ba9dc15bebade69 CarrySkipAdder.cpp\nce45d117e5a45c141acb0a305a0a6eaf30ccdc4b387a55bd5ab330672bf3c03a CarrySkipAdder.hpp\n227583b37ff516c46466fc5c918ee7605b0476c7e79e1b63b1fc28017046b0d2 CarrySkipAdderVariable.cpp\n1812c7c9c45afa80da1317afd4c2210c3583748daed3177e247887653392a938 CarrySkipAdderVariable.hpp\n1448c3a20470e843c0e3d8e81150aa7ae966b01c0324366ad6d09973635eae33 Dadda.cpp\nb409b894cf7e5dd8b1566ad65242d8204a3b10e09218cfdba51d6798420c3db4 Dadda.hpp\ne2de7008e12d31cf28e87ddcc91be289718e8c57104720d626d739224e17d237 GenMul.cpp\n9f7728b1956d663933ee8c2e36eedbfbf99e183f597155867c1b3ea7b518cfc7 GenMul.hpp\nca6978e216edf5aa647ec0ac3ce023db92fac8b0514ff9494e53594d0f6870ea GenMul_Emscripten.cpp\n59b5e2c157f082415d42add003c3f61c2a59ddcff55d4bce8b87b7769944c265 GenMul_Emscripten.hpp\nb6c44052471006782baa14f9ee553e9dd2befb5a48caaf3ea08ada0fa333a4b7 KoggeStoneAdder.cpp\n39decce3c706b8eafdd039498cb80147312dc279cf9ac722df1d7c5e44ca1eb1 KoggeStoneAdder.hpp\n8cd281edd6691072195380d088347b8080c00b9bdc903f9930b6f9b66343a532 LanderFischerAdder.cpp\n5a7fd4ccf6aba5d41505bf471d36050ebc082c671b174d26d2afe89d8ab1424f LanderFischerAdder.hpp\nbc20681185e8c41a7f23ff828021720d345a078c484211fe91f92bc0bffc69e2 ModuleConnector.cpp\ne1e43efd032bbdf2f7cde39e51a5ca384a6f7dbe2bbb2d60b2ce379cba5c5a61 ModuleConnector.hpp\n8ce42bf2a0b4a30bf2a9b1371c7e59d94c8b71af34319211923ebc0acae9f0d0 PartialProductGenerator.cpp\n0ec3b8793327e4b4e723a90196524386697346d2ab66595de11ae34a7ed16ae2 PartialProductGenerator.hpp\n049b30b6e146bee3b19fa21ee9d78e1fba8caf3bf38b0e4a99bc6e35110f8cef PartialProductGeneratorSigned.cpp\nb3fd13d4dc90c5708b5905c73a6437578e972e9929a8f5ef40d224277a72b3eb PartialProductGeneratorSigned.hpp\nb4d8f357fdd48208ab4dbec18a26d3dd8091c289f388f9bfe66dfe22793c005c RippleCarryAdder.cpp\n7a68cc632729d6a10a87e0d6750a3b6bc38ffea3c1adefbb6c04f5c81d6821fd RippleCarryAdder.hpp\n02ad7291a8c88d72769d019fcfcb63f20595eeb7f97880174f7080c433adc827 SerialPrefixAdder.cpp\n34c8850d96db902dcf9124b07b854739c311e1b984f27ae2cba2c97dac68b528 SerialPrefixAdder.hpp\nce0ae8c29d242e8eda47cc3e9ff06bb052ce1a90f3c8e0cf21ee13c63ddd30e9 VerilogGen.cpp\n60865ceaeae304e64ad944fb7b83fda22058c3e042b9e5ad1e2b35effcb79918 VerilogGen.hpp\n53e7bc2a228005f30c0349cd8f1e2f68498141e175ee1c140d3465823cfc44ec Wallace.cpp\nf00fd621015cfcf54cda85815dce343d8526c1fee0f7e22a9d98041195b72e1a Wallace.hpp\n421854807ffbed49e9450453ff4d7b2fb6cbcff53c1e8cca3299cc3c49d5bb6f Wallace5.cpp\nae5f879f4f6fda292e6394ba3e8df4a50b38d2c1c8f590fbf5752c2109deb398 Wallace5.hpp\n6cb775e3f48c2bc7fe32ff327217939a020f17c5087cc13d0faee3452517d768 component.cpp\n209f908d2f1fb45ff2ab578483ac59b4daf3db09b0623e939fe467d8c408a03c component.hpp\ne882c7b4ccf415774af64e8554a5d95bc8652feda0a36406f200e95b0228339b main.cpp\n287f82e4c4496d55e1bfa8679ee9e55e472790befa8efd8632594da9f222bcc6 partial.cpp\n0f936d14fa3aa23fae3171e401a240bea867f3be47b3cf7302cfabd3cb016a2a partial.hpp\n---------------------------------------------------------------------------------------------------*/
