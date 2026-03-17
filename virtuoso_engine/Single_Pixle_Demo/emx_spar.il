; ============================================================
; 1. MASTER WRAPPER FUNCTION
; ============================================================
procedure( run_full_rf_pixel_flow()
    let( (libName cellName schName dest_folder base_output 
          s_param_pattern s_param_file report_file log_file netlist_path)
        
        ; ----------------------------------------------------
        ; Configuration & File Management
        ; ----------------------------------------------------
        libName      = "mingTest"
        cellName     = "auto_layout_pixel_1"
        schName      = "auto_layout_pixel_1_emx"
        
        ; Create a dedicated destination folder
        dest_folder  = strcat(pwd() "/emx_results")
        unless( isDir(dest_folder) createDir(dest_folder) )
        
        ; Define paths using the destination folder
        base_output      = strcat(dest_folder "/" cellName)
        
        ; EMX target pattern with literal '%d'
        s_param_pattern  = strcat(base_output ".s%dp") 
        
        ; Expected final Touchstone file (3 external + 6 blackbox ports = 9 ports)
        s_param_file     = strcat(base_output ".s10p")   
        
        report_file      = strcat(base_output "_emx.pin")
        log_file         = strcat(base_output "_emx.log")
        netlist_path     = strcat(dest_folder "/netlist")
        
        printf("--- STARTING AUTOMATED RF FLOW ---\n")
        
        ; ----------------------------------------------------
        ; Execute Modular Flow
        ; ----------------------------------------------------
        ;rf_flow_run_emx(libName cellName dest_folder s_param_pattern report_file log_file)
        ;rf_flow_build_schematic(libName schName report_file s_param_file)
        ;rf_flow_create_netlist(netlist_path s_param_file)
        ;rf_flow_run_sim(netlist_path dest_folder)
        

		; ----------------------------------------------------
        ; Execute Modular Flow
        ; ----------------------------------------------------
        rf_flow_run_emx(libName cellName dest_folder s_param_pattern report_file log_file)
        ;rf_flow_build_schematic(libName schName report_file s_param_file)
        
        ; Replace the old netlist/sim functions with this single line:
        rf_flow_run_sim(dest_folder cellName s_param_file)
        printf("--- AUTOMATED RF FLOW COMPLETE ---\n")
    )
)

; ============================================================
; 2. EMX EXTRACTION FUNCTION (Corrected GDS Export)
; ============================================================
procedure( rf_flow_run_emx( libName cellName dest_folder s_param_pattern report_file log_file )
    let( (gds_file proc_file strmout_cmd emx_cmd sys_status bb_file bb_filename strmout_log)
        
        ; ----------------------------------------------------
        ; 1. Export GDSII using 'strmout' CLI
        ; ----------------------------------------------------
        printf(">> Exporting GDSII from Virtuoso...\n")
        gds_file = strcat(dest_folder "/" cellName ".gds")
        strmout_log = strcat(dest_folder "/strmout.log")
        
        ; Build the standard Cadence stream-out shell command
       strmout_cmd = sprintf(nil "strmout -library %s -topCell %s -view layout -strmFile %s -logFile %s -pinAttNum 51"
                              libName cellName gds_file strmout_log)
        
		;strmout_cmd = sprintf(nil "strmout -help")
        printf("Executing: %s\n" strmout_cmd)
        sys_status = system(strmout_cmd)
        
        if( sys_status != 0 || !isFile(gds_file) then 
            error("GDS export failed! Check %s for details." strmout_log) 
        )
        printf(">> GDS Export Complete: %s\n" gds_file)

        ; ----------------------------------------------------
        ; 2. Generate Blackbox Rules
        ; ----------------------------------------------------
        bb_filename = strcat(dest_folder "/blackbox_cells.txt")
        bb_file = outfile(bb_filename)
        if( !bb_file then error("Could not create blackbox cells file!") )
        fprintf(bb_file "^nmos_rf.*\n")
        fprintf(bb_file "^mimcap.*\n")
        close(bb_file)

        ; ----------------------------------------------------
        ; 3. Run EMX
        ; ----------------------------------------------------
        printf(">> Running EMX Extraction...\n")
        
        ; IMPORTANT: Provide the absolute path to your TSMC 65nm process file (.proc or .ltd)
        proc_file = "/disk/mcao53/Project65/proc65/RC_IRCX_CRN65G+_1P9M+UT-ALRDL_6X1Z1U_MiM_typical.proc" 
        
        ; Add 'cd %s && ' to the beginning of the string
        ; Add 'dest_folder' as the very first argument passed to sprintf
        emx_cmd = sprintf(nil "cd %s && /cae/apps/data/cadence-2025/installs/EMX20251/bin/emx %s %s %s --sweep 0 20e9 --sweep-stepsize 100e6 --device-cells-file %s -p PIX_OUT -p VSS -p GATE --message-file %s --log-time-level=0 --parallel 8 --recommended-memory --format touchstone2 -s %s --include-port-order --cadence-pins=51" 
                          dest_folder gds_file cellName proc_file bb_filename log_file s_param_pattern)

        printf("Executing: %s\n" emx_cmd)
        
        sys_status = system(emx_cmd)
        if( sys_status != 0 then error("EMX extraction failed! Check %s for details." log_file) )
        
        printf(">> EMX Complete. Results saved in %s\n" dest_folder)
    )
)

; ============================================================
; 3. SCHEMATIC GENERATION FUNCTION (For LVS / Visuals)
; ============================================================
procedure( rf_flow_build_schematic( libName schName report_file s_param_file )
    let( (in_port line tokens current_bb_port dev_map sch_cv 
          x_pos y_pos master inst stub_end wire net_name nport_master nport_inst)
          
        printf(">> Building LVS Schematic...\n")
        
        in_port = infile(report_file)
        if( !in_port then error("Could not open pin report: %s" report_file) )
        
        dev_map = makeTable("dev_map" nil) 
        current_bb_port = 4 ; BB ports start at 4
        
        while( gets(line in_port)
            line = buildString(parseString(line "\n\r") "") 
            if( pcreMatchp("^~x[0-9]+" line) then
                gets(line in_port) 
                tokens = parseString(line " \t")
                if( length(tokens) >= 3 then
                    let( (d_lib d_cell d_pin key)
                        d_lib  = nth(0 tokens)
                        d_cell = nth(1 tokens)
                        d_pin  = nth(2 tokens)
                        key    = strcat(d_lib "|" d_cell)
                        
                        if( !dev_map[key] then dev_map[key] = list() )
                        dev_map[key] = cons( list(d_pin current_bb_port) dev_map[key] )
                        current_bb_port = current_bb_port + 1
                    )
                )
            )
        )
        close(in_port)
        
        sch_cv = dbOpenCellViewByType(libName schName "schematic" "schematic" "w")
        x_pos = 0.0
        y_pos = 0.0
        
        foreach( key dev_map
            let( (lib_cell_list d_lib d_cell pin_data)
                lib_cell_list = parseString(key "|")
                d_lib = nth(0 lib_cell_list)
                d_cell = nth(1 lib_cell_list)
                pin_data = dev_map[key]
                
                master = dbOpenCellViewByType(d_lib d_cell "symbol")
                if( master then
                    inst = schCreateInst(sch_cv master nil list(x_pos y_pos) "R0")
                    foreach( p_info pin_data
                        let( (pin_name port_num master_term)
                            pin_name = nth(0 p_info)
                            port_num = nth(1 p_info)
                            net_name = sprintf(nil "EMX_PORT_%d" port_num)
                            
                            master_term = dbFindTermByName(master pin_name)
                            if( master_term then
                                stub_end = list(x_pos - 0.5 y_pos + 0.5) 
                                wire = schCreateWire(sch_cv "draw" "full" list(list(x_pos y_pos) stub_end) 0.0 0.0 0.0)
                                schCreateWireLabel(sch_cv car(wire) stub_end net_name "centerLeft" "R0" "stick" 0.0625 nil)
                            )
                        )
                    )
                    x_pos = x_pos + 3.0 
                )
            )
        )
        
        nport_master = dbOpenCellViewByType("analogLib" "nport" "symbol")
        if( nport_master then
            nport_inst = schCreateInst(sch_cv nport_master "EMX_DATA" list(x_pos y_pos) "R0")
            dbCreateProp(nport_inst "dataFile" "string" s_param_file)
        )
        
        schCheck(sch_cv)
        dbSave(sch_cv)
        dbClose(sch_cv)
        printf(">> Schematic %s saved.\n" schName)
    )
)

; ============================================================
; 4. NETLIST & SIMULATE FUNCTION (Updated for 3-Terminal MIM)
; ============================================================
procedure( rf_flow_run_sim( dest_folder cellName s_param_file )
    let( (vctrl_vals netlist_path result_dir out_port sim_cmd idx smith_win)
        printf(">> Starting Direct Spectre Simulation...\n")
        
        vctrl_vals = list(1.0 0.0)
        idx = 0 ; Initialize our file naming index
		smith_win = awvCreatePlotWindow()
        awvSetCurrentWindow(smith_win)
        
        foreach( v vctrl_vals
            printf(">> Simulating Run %d with V_ctrl = %g V...\n" idx v)
            
            ; 1. Define unique paths using the INTEGER index
            netlist_path = sprintf(nil "%s/tb_%s_run%d.scs" dest_folder cellName idx)
            result_dir   = sprintf(nil "%s/psf_run%d" dest_folder idx)
            
            ; 2. Generate the Netlist
            out_port = outfile(netlist_path)
            fprintf(out_port "// Automated EMX Testbench for Run %d (V_ctrl = %gV)\n" idx v)
            fprintf(out_port "simulator lang=spectre\n\n")
            
            ; Models
            fprintf(out_port "include \"/disk/TSMC65GP_PDK/MSRF_General_Purpose_Plus/PDK/CadenceOA/tn65cmsp018k3_1_0c/models/spectre/crn65gplus_2d5_lk_v1d0.scs\" section=tt_rfmos\n\n")
            fprintf(out_port "include \"/disk/TSMC65GP_PDK/MSRF_General_Purpose_Plus/PDK/CadenceOA/tn65cmsp018k3_1_0c/models/spectre/crn65gplus_2d5_lk_v1d0.scs\" section=tt_rfmim\n\n")
            
            ; Inject the actual floating point voltage into the physics simulation
            fprintf(out_port "parameters v_ctrl=%g\n\n" v)
            
            ; --- CHANGE 1: EMX NPORT MAPPING ---
            ; The order here MUST match the bottom 10 lines of your EMX report:
            ; 1:GATE, 2:PIX_OUT, 3:VSS, 4:~x0000(B), 5:~x0001(D), 6:~x0002(G), 7:~x0003(S), 8:~x0004(PLUS), 9:~x0005(MINUS), 10:~x0006(BULK)
            ;fprintf(out_port "emx_blk (gate pix_out vss nmos_b nmos_d nmos_g nmos_s mim_p mim_m mim_b) nport file=\"%s\"\n\n" s_param_file)
			fprintf(out_port "emx_blk (gate 0 pix_out 0 vss 0 nmos_b 0 nmos_d 0 nmos_g 0 nmos_s 0 mim_p 0 mim_m 0 mim_b 0) nport file=\"%s\"\n\n" s_param_file)            
            
			fprintf(out_port "// Re-inserting TSMC Models\n")
            fprintf(out_port "M0 (nmos_d nmos_g nmos_s nmos_b) nmos_rf wr=3u lr=80n nr=4\n")
            
            ; --- CHANGE 2: 3-TERMINAL MIM INSTANCE ---
            ; Adding 'mim_b' as the third parameter to represent the bulk/substrate
            fprintf(out_port "C0 (mim_p mim_m mim_b) mimcap_um_sin_rf wt=15u lt=15u\n\n")
            
            fprintf(out_port "// Stimulus\n")
            fprintf(out_port "v_vss  (vss 0) vsource dc=0\n")
            
            ; --- CHANGE 3: GROUNDING THE NEW BULK ---
            ; Replaced the old 'emx_gnd' with 'mim_b' tied to 0V (GND)
            fprintf(out_port "v_mimb (mim_b 0) vsource dc=0\n")
            
            fprintf(out_port "v_gate (gate 0) vsource dc=v_ctrl\n")
            fprintf(out_port "p1 (pix_out 0) port r=50 num=1\n\n")
            
            ; --- NEW MODIFICATION ---
            ; Define the output path for the Touchstone file
            s1p_path = sprintf(nil "%s/%s_run%d.s1p" dest_folder cellName idx)
            
            ; Tell Spectre to dump the S-parameters directly to this file
            fprintf(out_port "sp_sim sp ports=[p1] start=1G stop=20G step=100M file=\"%s\" datafmt=touchstone\n" s1p_path)
            close(out_port)
            
            ; 3. Run Spectre via command line
            sim_cmd = sprintf(nil "spectre %s -raw %s" netlist_path result_dir)
            printf("Executing: %s\n" sim_cmd)
            system(sim_cmd)
            
			if( isDir(result_dir) then
                printf(">> Simulation %d complete. Data saved to %s_run%d.s1p\n" idx cellName idx)
            else
                warn("Simulation failed or result directory not found for Run %d!" idx)
            )		
            
            ; 4. Load the data back into Virtuoso and Plot
            ;if( isDir(result_dir) then
            ;    openResults(result_dir)
            ;    selectResult('sp)
            ;    
            ;    ; 1. Plot S11 in the default rectangular window (Magnitude/Phase)
            ;    plot(sp(1 1) ?expr list(sprintf(nil "S11_Rect_Vctrl_%gV" v)))
            ;    
            ;    ; 2. Plot S11 into the dedicated Smith Chart window
            ;    ; Using awvPlotWaveform with ?graphType forces the Smith Chart rendering in ViVA
            ;    awvPlotWaveform( smith_win 
            ;        list( sp(1 1) ) 
            ;        ?expr list( sprintf(nil "Z_Smith_Vctrl_%gV" v) ) 
            ;        ?graphType "impedance" 
            ;    )
            ;    
            ;else
            ;    warn("Simulation failed or result directory not found for Run %d!" idx)
            ;)
            
            idx = idx + 1
        )
        printf(">> All Simulations Complete!\n")
    )
)


; ============================================================
; 4. NETLIST GENERATION FUNCTION (For Spectre)
; ============================================================
procedure( rf_flow_create_netlist( netlist_path s_param_file )
    let( (out_port)
        printf(">> Generating Spectre Netlist...\n")
        
        out_port = outfile(netlist_path)
        fprintf(out_port "// Automated EMX Testbench\n")
        fprintf(out_port "simulator lang=spectre\n\n")
        
        ; *** UPDATE THIS PATH TO YOUR TSMC 65NM MODELS ***
        fprintf(out_port "include \"/disk/TSMC65GP_PDK/MSRF_General_Purpose_Plus/PDK/CadenceOA/tn65cmsp018k3_1_0c/models/spectre/crn65gplus_2d5_lk_v1d0.scs\" section=tt\n\n")
        
        fprintf(out_port "parameters v_ctrl=1.0\n\n")
        
        fprintf(out_port "emx_blk (pix_out vss gate  nmos_b nmos_d nmos_g nmos_s  mim_m mim_p  emx_gnd) nport file=\"%s\"\n\n" s_param_file)
        
        fprintf(out_port "// Re-inserting TSMC Models\n")
        fprintf(out_port "M0 (nmos_d nmos_g nmos_s nmos_b) nmos_rf w=3u l=80n nfing=4\n")
        fprintf(out_port "C0 (mim_p mim_m) mimcap w=15u l=15u\n\n")
        
        fprintf(out_port "// Stimulus\n")
        fprintf(out_port "v_vss  (vss 0) vsource dc=0\n")
        fprintf(out_port "v_egnd (emx_gnd 0) vsource dc=0\n")
        fprintf(out_port "v_gate (gate 0) vsource dc=v_ctrl\n")
        fprintf(out_port "p1 (pix_out 0) port r=50 num=1\n\n")
        
        fprintf(out_port "sp_sim sp ports=[p1] start=1G stop=20G step=100M\n")
        close(out_port)
        printf(">> Netlist created at %s\n" netlist_path)
    )
)


