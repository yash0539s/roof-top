<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0">
    <rules key="root">
      <rule key="asb" label="Asbestos" filter="&quot;pred_class&quot; = 'ASBESTOS'" symbol="0"/>
      <rule key="met" label="Metal" filter="&quot;pred_class&quot; = 'METAL'" symbol="1"/>
      <rule key="rcc" label="RCC" filter="&quot;pred_class&quot; = 'RCC'" symbol="2"/>
      <rule key="til" label="Tiled" filter="&quot;pred_class&quot; = 'TILED'" symbol="3"/>
      <rule key="err" label="Misclassified (field label differs)" filter="&quot;correct&quot; = 0" symbol="4"/>
    </rules>
    <symbols>
      <symbol type="fill" name="0" alpha="0.75" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="148,103,189,255"/>
            <Option name="outline_color" type="QString" value="40,40,40,255"/>
            <Option name="outline_width" type="QString" value="0.3"/>
            <Option name="style" type="QString" value="solid"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="0.75" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="31,119,180,255"/>
            <Option name="outline_color" type="QString" value="40,40,40,255"/>
            <Option name="outline_width" type="QString" value="0.3"/>
            <Option name="style" type="QString" value="solid"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="0.75" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="160,160,160,255"/>
            <Option name="outline_color" type="QString" value="40,40,40,255"/>
            <Option name="outline_width" type="QString" value="0.3"/>
            <Option name="style" type="QString" value="solid"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="3" alpha="0.75" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="214,96,77,255"/>
            <Option name="outline_color" type="QString" value="40,40,40,255"/>
            <Option name="outline_width" type="QString" value="0.3"/>
            <Option name="style" type="QString" value="solid"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="4" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="0,0,0,0"/>
            <Option name="outline_color" type="QString" value="227,26,28,255"/>
            <Option name="outline_width" type="QString" value="1.2"/>
            <Option name="style" type="QString" value="no"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fieldName="UID" fontSize="7" textColor="20,20,20,255"/>
      <text-buffer bufferDraw="1" bufferSize="0.7" bufferColor="255,255,255,255"/>
    </settings>
  </labeling>
</qgis>
