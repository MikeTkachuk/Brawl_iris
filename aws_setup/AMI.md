### EC2 Image Builder setup

1) Create software components from the files provided in ./
2) Create image recipe, choose ami-07024bf15a5ac31ca as a base AMI
3) Leave builder working directory as is (/tmp by default)
4) Add two components in the following order:
   - conda (conda_component.txt)
   - env and packages (dependencies_component.txt)
5) Create image builder IAM role https://docs.aws.amazon.com/imagebuilder/latest/userguide/image-builder-setting-up.html
6) Create infrastructure configuration by specifying:
   - the role you've created
   - g4dn.xlarge instance
7) Define a new image builder pipeline and specify:
   - type manual
   - the recipe you've created
   - the infrastructure configuration you've created
   - leave everything else in defaults
8) Run pipeline and make sure the image is created and registered in EC2.AMIs
