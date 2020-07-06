C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.vert -o base_vert.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.frag -o base_frag.spv

C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.vert -DHAS_UV_SET0=1 -o base_vert_TEX.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.frag -DHAS_UV_SET0=1 -o base_frag_TEX.spv

C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.vert -DHAS_NORMALS=1 -o base_vert_N.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.frag -DHAS_NORMALS=1 -o base_frag_N.spv

C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.vert -DHAS_TANGENT=1 -o base_vert_TN.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.frag -DHAS_TANGENT=1 -o base_frag_TN.spv

C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.vert -DHAS_UV_SET0=1 -DHAS_NORMALS=1 -o base_vert_TEX_N.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.frag -DHAS_UV_SET0=1 -DHAS_NORMALS=1 -o base_frag_TEX_N.spv

C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.vert -DHAS_UV_SET0=1 -DHAS_TANGENT=1 -o base_vert_TEX_TN.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.frag -DHAS_UV_SET0=1 -DHAS_TANGENT=1 -o base_frag_TEX_TN.spv

C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.vert -DHAS_NORMALS=1 -DHAS_SKIN_SET_0=1 -o base_vert_N_SKIN.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe base.frag -DHAS_NORMALS=1 -DHAS_SKIN_SET_0=1 -o base_frag_N_SKIN.spv

C:\work\vulkan\1.2.141.2\Bin\glslc.exe skybox.vert -o skybox_vert.spv
C:\work\vulkan\1.2.141.2\Bin\glslc.exe skybox.frag -o skybox_frag.spv

pause