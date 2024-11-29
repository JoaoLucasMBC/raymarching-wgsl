const THREAD_COUNT = 16;
const PI = 3.1415927f;
const MAX_DIST = 1000.0;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> shapesb : array<shape>;

@group(2) @binding(1)
  var<storage, read_write> shapesinfob : array<vec4f>;

struct shape {
  transform : vec4f, // xyz = position
  radius : vec4f, // xyz = scale, w = global scale
  rotation : vec4f, // xyz = rotation
  op : vec4f, // x = operation, y = k value, z = repeat mode, w = repeat offset
  color : vec4f, // xyz = color
  animate_transform : vec4f, // xyz = animate position value (sin amplitude), w = animate speed
  animate_rotation : vec4f, // xyz = animate rotation value (sin amplitude), w = animate speed
  quat : vec4f, // xyzw = quaternion
  transform_animated : vec4f, // xyz = position buffer
};

struct march_output {
  color : vec3f,
  depth : f32,
  outline : bool,
};

fn op_smooth_union(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);

  var h = clamp(0.5 + 0.5 * (d2 - d1) / k_eps, 0.0, 1.0);

  // Mix
  var d_res = mix(d2, d1, h) - k_eps * h * (1.0 - h);
  var col_res = mix(col2, col1, h);

  return vec4f(col_res, d_res);
}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);

  var h = clamp(0.5 - 0.5 * (d2 + d1) / k_eps, 0.0, 1.0);

  // Mix
  var d_res = mix(d2, -d1, h) + k_eps * h * (1.0 - h);
  var col_res = mix(col2, col1, h);

  return vec4f(col_res, d_res);
}

fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);

  var h = clamp(0.5 - 0.5 * (d2 - d1) / k_eps, 0.0, 1.0);

  // Mix
  var d_res = mix(d2, d1, h) + k_eps * h * (1.0 - h);
  var col_res = mix(col2, col1, h);

  return vec4f(col_res, d_res);
}

fn op(op: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  // union
  if (op < 1.0)
  {
    return op_smooth_union(d1, d2, col1, col2, k);
  }

  // subtraction
  if (op < 2.0)
  {
    return op_smooth_subtraction(d1, d2, col1, col2, k);
  }

  // intersection
  return op_smooth_intersection(d2, d1, col2, col1, k);
}

fn repeat(p: vec3f, offset: vec3f) -> vec3f
{
  return modc(p + 0.5 * offset, offset) - 0.5 * offset;
}

fn transform_p(p: vec3f, option: vec2f) -> vec3f
{
  // normal mode
  if (option.x <= 1.0)
  {
    return p;
  }

  // return repeat / mod mode
  return repeat(p, vec3f(option.y));
}

fn scene(p: vec3f) -> vec4f // xyz = color, w = distance
{
    var d = mix(100.0, p.y, uniforms[17]);

    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);

    var all_objects_count = spheresCount + boxesCount + torusCount;
    var result = vec4f(vec3f(1.0), d) ;

    // mendelbulb
    if (uniforms[18] > 0.0)
    {
      var bulb = sdf_mandelbulb(p);
      result = vec4f(vec3f(bulb.y),  bulb.x);
    }
    else if (uniforms[19] > 0.0)
    {
      var weird = sdf_weird_thing(p, uniforms[19]);
      result = vec4f(vec3f(1.0), weird);
    }
    else
    {
      for (var i = 0; i < all_objects_count; i = i + 1)
      {
        // get shape and shape order (shapesinfo)
        // shapesinfo has the following format:
        // x: shape type (0: sphere, 1: box, 2: torus)
        // y: shape index
        // order matters for the operations, they're sorted on the CPU side
        var info = shapesinfob[i];
        var shape = shapesb[i32(info.y)];

        var p_ = transform_p(p - shape.transform_animated.xyz, vec2f(shape.op.z, shape.op.w));

        if (info.x < 1.0)
        {
          d = sdf_sphere(p_, shape.radius, shape.quat);
        }
        else if (info.x < 2.0)
        {
          d = sdf_round_box(p_, shape.radius.xyz + shape.radius.w, shape.radius.w, shape.quat);
        }
        else
        {
          d = sdf_torus(p_, shape.radius.xy, shape.quat);
        }

        // call op function with the shape operation
        result = op(shape.op.x, d, result.w, shape.color.xyz, result.xyz, shape.op.y);
      }
    }

    return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
{
  var max_marching_steps = i32(uniforms[5]);
  var EPSILON = uniforms[23];

  var depth = 0.0;
  var min_d = MAX_DIST;
  var color = vec3f(0.0);
  var march_step = uniforms[22];

  var has_outline = uniforms[26];
  var outline_w = uniforms[27];
  var outline_color = vec3f(1.0) * uniforms[28];
  
  for (var i = 0; i < max_marching_steps; i = i + 1)
  {
      // raymarch algorithm
      // call scene function and march
      var ro_ = ro + rd * depth;
      
      var result = scene(ro_);

      color = result.xyz;

      min_d = min(result.w, min_d);

      // if the depth is greater than the max distance or the distance is less than the epsilon, break
      if (depth > MAX_DIST)
      {
        break;
      }

      if (result.w < EPSILON)
      {
        return march_output(color, depth, false);
      }

      depth = depth + result.w;
  }

  // check if we should draw the outline of something
  if (has_outline > 0.0 && min_d < outline_w)
  {
    return march_output(outline_color, depth, true);
  }

  return march_output(vec3f(0.0), MAX_DIST, false);
}

fn get_normal(p: vec3f) -> vec3f
{
  var e = 0.0001;
  return normalize(vec3(
    scene(vec3(p.x + e, p.y, p.z)).w - scene(vec3(p.x - e, p.y, p.z)).w,
    scene(vec3(p.x, p.y + e, p.z)).w - scene(vec3(p.x, p.y - e, p.z)).w,
    scene(vec3(p.x, p.y, p.z + e)).w - scene(vec3(p.x, p.y, p.z - e)).w
  ));
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32
{
  var res = 1.0;
  var t = tmin;

  for (var i = 0; i < 256 && t < tmax; i = i + 1)
  {
    var h = scene(ro + rd * t).w;
    res = min(res, h/(k*t));
    t += clamp(h, 0.005, 0.5);
    if (res < -1.0 || t > tmax)
    {
      break;
    }
  }
  res = max(res, -1.0);
  return 0.25 * (1.0 + res)*(1.0 + res)*(2.0 - res);
}

fn get_AO(current: vec3f, normal: vec3f) -> f32
{
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i = i + 1)
  {
    var h = 0.001 + 0.15 * f32(i) / 4.0;
    var d = scene(current + h * normal).w;
    occ += (h - d) * sca;
    sca *= 0.95;
  }

  return clamp( 1.0 - 2.0 * occ, 0.0, 1.0 ) * (0.5 + 0.5 * normal.y);
}

fn get_ambient_light(light_pos: vec3f, sun_color: vec3f, rd: vec3f) -> vec3f
{
  var backgroundcolor1 = int_to_rgb(i32(uniforms[12]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[29]));
  var backgroundcolor3 = int_to_rgb(i32(uniforms[30]));
  
  var ambient = backgroundcolor1 - rd.y * rd.y * 0.5;
  ambient = mix(ambient, 0.85 * backgroundcolor2, pow(1.0 - max(rd.y, 0.0), 4.0));

  var sundot = clamp(dot(rd, normalize(vec3f(light_pos))), 0.0, 1.0);
  var sun = 0.25 * sun_color * pow(sundot, 5.0) + 0.25 * vec3f(1.0,0.8,0.6) * pow(sundot, 64.0) + 0.2 * vec3f(1.0,0.8,0.6) * pow(sundot, 512.0);
  ambient += sun;
  ambient = mix(ambient, 0.68 * backgroundcolor3, pow(1.0 - max(rd.y, 0.0), 16.0));

  return ambient;
}

fn get_light(current: vec3f, obj_color: vec3f, rd: vec3f) -> vec3f
{
  var light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
  var sun_color = int_to_rgb(i32(uniforms[16]));
  var ambient = get_ambient_light(light_position, sun_color, rd);
  var normal = get_normal(current);

  // calculate light based on the normal
  // if the object is too far away from the light source, return ambient light
  if (length(current) > uniforms[20] + uniforms[8])
  {
    return ambient;
  }

  var light_direction = normalize(light_position - current);
  
  // do the diffuse
  var diff = max(0.0, dot(normal, light_direction));
  var diff_color = diff * obj_color * sun_color;

  // do shadow
  var shadow = get_soft_shadow(current + normal * 0.001, light_direction, uniforms[24], uniforms[25], uniforms[21]);
  diff_color *= shadow;

  // do ambient occlusion
  var occ = get_AO(current, normal);

  var color = ambient * obj_color;
  color += diff_color;
  color *= occ;

  return clamp(color, vec3f(0.0), vec3f(1.0));
}

fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}

// why amplitude? Font: GPT
fn animate(val: vec3f, amplitude: vec3f, time_scale: f32, offset: f32) -> vec3f {
    var angle = time_scale * offset;
    var x = amplitude.x * sin(angle);
    var z = amplitude.z * cos(angle);
    var y = amplitude.y * cos(angle);

    return val + vec3f(x, y, z);
}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn preprocess(@builtin(global_invocation_id) id : vec3u)
{
  var time = uniforms[0];
  var spheresCount = i32(uniforms[2]);
  var boxesCount = i32(uniforms[3]);
  var torusCount = i32(uniforms[4]);
  var all_objects_count = spheresCount + boxesCount + torusCount;

  if (id.x >= u32(all_objects_count))
  {
    return;
  }

  var shape = shapesb[i32(id.x)];

  // update animation transform
  var animated_transform = animate(shape.transform.xyz, shape.animate_transform.xyz, shape.animate_transform.w, time);

  var animated_rotation = animate(shape.rotation.xyz, shape.animate_rotation.xyz, shape.animate_rotation.w, time);

  shape.transform_animated = vec4f(animated_transform, shape.transform.w);  
  shape.quat = quaternion_from_euler(animated_rotation);

  // I HATE THIS SHIT!!
  shapesb[i32(id.x)] = shape;
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
  // unpack data
  var fragCoord = vec2f(f32(id.x), f32(id.y));
  var rez = vec2(uniforms[1]);
  var time = uniforms[0];

  // camera setup
  var lookfrom = vec3(uniforms[6], uniforms[7], uniforms[8]);
  var lookat = vec3(uniforms[9], uniforms[10], uniforms[11]);
  var camera = set_camera(lookfrom, lookat, 0.0);
  var ro = lookfrom;

  // get ray direction
  var uv = (fragCoord - 0.5 * rez) / rez.y;
  uv.y = -uv.y;
  var rd = camera * normalize(vec3(uv, 1.0));

  // call march function and get the color/depth
  var march_result = march(ro, rd);
  var depth = march_result.depth;

  var color = vec3f(0.0);
  if (march_result.outline)
  {
    color = march_result.color;
  }
  else if (depth < MAX_DIST)
  {
    // move ray based on the depth
    var p = ro + rd * march_result.depth;
    // get light
    color = get_light(p, march_result.color, rd);
  }
  else 
  {
    // get ambient light se não bateu em nada
    color = get_ambient_light(vec3f(uniforms[13], uniforms[14], uniforms[15]), int_to_rgb(i32(uniforms[16])), rd);
  }
  
  // display the result
  color = linear_to_gamma(color);
  fb[mapfb(id.xy, uniforms[1])] = vec4(color, 1.0);
}