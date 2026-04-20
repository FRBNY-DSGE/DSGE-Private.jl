% Collect blocks that do not need to be updatred

F1Xnext = F1_aux(1:grid.numstates-grid.os,:);
F21next = zeros(grid.os_agg,grid.numstates-grid.os);
F31next = zeros(grid.os_process,grid.numstates-grid.os);
F32next = zeros(grid.os_process,grid.os_agg);
F33next = eye(grid.os_process);
F3Xnext = [F31next,F32next,F33next];
F41next = zeros(grid.numcontrols-grid.oc,grid.numstates-grid.os);
F42next = F1_aux(grid.numstates+1:end-grid.oc,end-grid.os+1:end-grid.os_process);
F43next = zeros(grid.numcontrols-grid.oc,grid.os_process);
F4Xnext = [F41next,F42next,F43next];
F51next = zeros(grid.oc_summary,grid.numstates-grid.os);
F53next = zeros(grid.oc_summary,grid.os_process);
F5Xnext = F1_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,:);
F61next = zeros(grid.oc_agg,grid.numstates-grid.os);

% F22next, F23next, F52next, F62next, F63next <= need to be updated

F1Ynext = F2_aux(1:grid.numstates-grid.os,:);
F24next = zeros(grid.os_agg,grid.numcontrols-grid.oc);
F3Ynext = zeros(grid.os_process,grid.numcontrols);
F4Ynext = F2_aux(grid.numstates+1:end-grid.oc,:);
F5Ynext = F2_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,:);
F64next = F2_aux(end-grid.oc_agg+1:end,1:end-grid.oc);

% F24next = F2_aux(grid.numstates-grid.os+1:grid.numstates-grid.os_process,1:end-grid.oc);
% F3Ynext = F2_aux(grid.numstates-grid.os_process+1:grid.numstates,:);
% F44next = F2_aux(grid.numstates+1:end-grid.oc,1:end-grid.oc);
% F45next = F2_aux(grid.numstates+1:end-grid.oc,end-grid.oc+1:end-grid.oc_agg);
% F54next = F2_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,1:end-grid.oc);
% F55next = F2_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,end-grid.oc+1:end-grid.oc+grid.oc_summary);


% F25next, F26next, F65next, F66next <= need to be updated

F1X     = F3_aux(1:grid.numstates-grid.os,:);
F21     = zeros(grid.os_agg,grid.numstates-grid.os);
F31     = zeros(grid.os_process,grid.numstates-grid.os);
F32     = zeros(grid.os_process,grid.os_agg);
F4X     = F3_aux(grid.numstates+1:end-grid.oc,:);
F51     = F3_aux(end-grid.oc+1:end-grid.oc_agg,1:grid.numstates-grid.os);
F5X     = F3_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,:);
F61     = F3_aux(end-grid.oc_agg+1:end,1:grid.numstates-grid.os);

% F21     = F3_aux(grid.numstates-grid.os+1:grid.numstates-grid.os+grid.os_agg,1:grid.numstates-grid.os);

% F22, F23, F33, F52, F53, F62, F63 <= need to be updated

F1Y     = F4_aux(1:grid.numstates-grid.os,:);
F24     = F4_aux(grid.numstates-grid.os+1:grid.numstates-grid.os+grid.os_agg,1:end-grid.oc);
F3Y     = zeros(grid.os_process,grid.numcontrols);
F4Y     = F4_aux(grid.numstates+1:end-grid.oc,:);
F54     = zeros(grid.oc_summary,grid.numcontrols-grid.oc);
F5Y     = F4_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,:);
F64     = F4_aux(end-grid.oc_agg+1:end,1:end-grid.oc);

% F25     = F4_aux(grid.numstates-grid.os+1:grid.numstates-grid.os+grid.os_agg,end-grid.oc+1:end-grid.oc_agg);


% F25, F26, F55, F56, F65, F66 <= need to be updated

Jacob_base.F1_aux = F1_aux;
Jacob_base.F2_aux = F2_aux;
Jacob_base.F3_aux = F3_aux;
Jacob_base.F4_aux = F4_aux;

% Jacob_ZLB_QE_compare.F1_aux = F1_ZLB_aux;
% Jacob_ZLB_QE_compare.F2_aux = F2_ZLB_aux;
% Jacob_ZLB_QE_compare.F3_aux = F3_ZLB_aux;
% Jacob_ZLB_QE_compare.F4_aux = F4_ZLB_aux;

Jacob_base.F1Xnext = F1Xnext;
Jacob_base.F21next = F21next;
Jacob_base.F31next = F31next;
Jacob_base.F32next = F32next;
Jacob_base.F33next = F33next;
Jacob_base.F3Xnext = F3Xnext;
Jacob_base.F41next = F41next;
Jacob_base.F42next = F42next;
Jacob_base.F43next = F43next;
Jacob_base.F4Xnext = F4Xnext;
Jacob_base.F51next = F51next;
Jacob_base.F53next = F53next;
Jacob_base.F61next = F61next;
Jacob_base.F1Ynext = F1Ynext;
Jacob_base.F24next = F24next;
Jacob_base.F3Ynext = F3Ynext;
Jacob_base.F4Ynext = F4Ynext;
Jacob_base.F5Xnext = F5Xnext;
Jacob_base.F5Ynext = F5Ynext;
Jacob_base.F64next = F64next;
Jacob_base.F1X     = F1X;
Jacob_base.F21     = F21;
Jacob_base.F31     = F31;
Jacob_base.F32     = F32;
Jacob_base.F4X     = F4X;
Jacob_base.F51     = F51;
Jacob_base.F61     = F61;
Jacob_base.F1Y     = F1Y;
Jacob_base.F24     = F24;
Jacob_base.F3Y     = F3Y;
Jacob_base.F4Y     = F4Y;
Jacob_base.F54     = F54;
Jacob_base.F5X     = F5X;
Jacob_base.F5Y     = F5Y;
Jacob_base.F64     = F64;


switch(param.adjust)
	case('G')
		% Jacob_base_LT = Jacob_base;
		% save('Jacob_base_LT','Jacob_base_LT');
		Jacob_base_G_tvcopula = Jacob_base;
		save('Jacob_base_G_tvcopula','Jacob_base_G_tvcopula');
	case('LT')
		% Jacob_base_G = Jacob_base;
		% save('Jacob_base_G','Jacob_base_G');
		Jacob_base_LT_tvcopula = Jacob_base;
		save('Jacob_base_LT_tvcopula','Jacob_base_LT_tvcopula');
end