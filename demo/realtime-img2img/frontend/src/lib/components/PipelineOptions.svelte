<script lang="ts">
  import type { Fields } from '$lib/types';
  import { FieldType } from '$lib/types';
  import InputRange from './InputRange.svelte';
  import SeedInput from './SeedInput.svelte';
  import TextArea from './TextArea.svelte';
  import Checkbox from './Checkbox.svelte';
  import Selectlist from './Selectlist.svelte';
  import { pipelineValues } from '$lib/store';

  export let pipelineParams: Fields;

  $: advanceOptions = Object.values(pipelineParams)?.filter(
    (e) => e?.hide == true && e?.disabled !== true
  );
  $: featuredOptions = Object.values(pipelineParams)?.filter((e) => e?.hide !== true);
</script>

<div class="flex flex-col gap-3">
  <div class="grid grid-cols-1 items-center gap-3">
    {#if featuredOptions}
      {#each featuredOptions as params}
        {#if params.field === FieldType.RANGE}
          <InputRange {params} bind:value={$pipelineValues[params.id]}></InputRange>
        {:else if params.field === FieldType.SEED}
          <SeedInput {params} bind:value={$pipelineValues[params.id]}></SeedInput>
        {:else if params.field === FieldType.TEXTAREA}
          <TextArea {params} bind:value={$pipelineValues[params.id]}></TextArea>
        {:else if params.field === FieldType.CHECKBOX}
          <Checkbox {params} bind:value={$pipelineValues[params.id]}></Checkbox>
        {:else if params.field === FieldType.SELECT}
          <Selectlist {params} bind:value={$pipelineValues[params.id]}></Selectlist>
        {/if}
      {/each}
    {/if}
  </div>
  {#if advanceOptions && advanceOptions.length > 0}
    <details>
      <summary class="cursor-pointer font-medium">Advanced Options</summary>
      <div
        class="grid grid-cols-1 items-center gap-3 {Object.values(pipelineParams).length > 5
          ? 'sm:grid-cols-2'
          : ''}"
      >
        {#each advanceOptions as params}
          {#if params.field === FieldType.RANGE}
            <InputRange {params} bind:value={$pipelineValues[params.id]}></InputRange>
          {:else if params.field === FieldType.SEED}
            <SeedInput {params} bind:value={$pipelineValues[params.id]}></SeedInput>
          {:else if params.field === FieldType.TEXTAREA}
            <TextArea {params} bind:value={$pipelineValues[params.id]}></TextArea>
          {:else if params.field === FieldType.CHECKBOX}
            <Checkbox {params} bind:value={$pipelineValues[params.id]}></Checkbox>
          {:else if params.field === FieldType.SELECT}
            <Selectlist {params} bind:value={$pipelineValues[params.id]}></Selectlist>
          {/if}
        {/each}
      </div>
    </details>
  {/if}
</div>
